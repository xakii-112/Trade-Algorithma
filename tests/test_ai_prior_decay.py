"""test_ai_prior_decay.py — Tur-25/WP-A (2026-08-10): zaman-ağırlıklı (üstel unutmalı)
posterior.

Kapsam:
  A1 (nihai/ai/learning.py) — `update_from_predictions`'a EK: `state["strategies_decayed"]`
    (üstel-unutmalı posterior) + `state["n_unparsed"]`; `beta_posterior` artık REEL sayım
    kabul eder.
  A2 (nihai/ai/decision.py) — `_prior_from_state`'in opt-in tüketimi: `AI_PRIOR_DECAY` env'i
    AÇIKSA `strategies_decayed`'i, kapalıyken (varsayılan) yalnız `strategies`'i okur.

ÖLÇÜLMÜŞ ARKA PLAN (varsayım değil): docs/AI_DENETIM_RAPORU_2026-07-26.md §E.4 — karar
motoru bayat bir yan-koşullu SHORT üstünlüğünü (geçmiş DOWN rejiminin artefaktı) prior alıp
yükselen piyasada SHORT girip kaybetti (net −%0.50, taban-göre −1.75 puan). Bu dosya o
mekanizmaya karşı eklenen düzeltmenin (a) varsayılan davranışı BOZMADIĞINI, (b) doğru
yönde çalıştığını, (c) kanıt-hacmi eşiklerini sessizce gevşetmediğini kilitler.

Konvansiyon: Türkçe test adları; `tmp_path`/`monkeypatch` ile izolasyon (kök `conftest.py`
zaten NIHAI_DATA_DIR/NIHAI_LOG_DIR'i geçici dizine yönlendiriyor — üretim dosyalarına
ASLA yazılmaz)."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta

import pytest

from nihai.ai import decision, episodic, learning

FIELDS = ["logged_at", "symbol", "interval", "bar_ts", "direction", "action", "confidence",
          "prob_cal", "pwin", "pwin_side", "price", "horizon_bars",
          "resolved_at", "fwd_price", "hit", "fwd_ret_pct"]


@pytest.fixture(autouse=True)
def _izole(tmp_path, monkeypatch):
    """learning.py + decision.py testleri için ORTAK izolasyon (mevcut test_ai_learning.py/
    test_ai_decision.py desenlerinin birleşimi): dosya yolları tmp_path'e yönlenir, bu
    dosyanın ilgilendiği TÜM env değişkenleri temiz başlar."""
    monkeypatch.setattr(learning, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(learning, "LEARNING_STATE", tmp_path / "ai_learning_state.json")
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "ai_decisions.jsonl")
    monkeypatch.setattr(episodic, "append_episode", lambda **kw: None)
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)
    monkeypatch.delenv("AI_PRIOR_DECAY", raising=False)
    monkeypatch.delenv("AI_PRIOR_HALF_LIFE_DAYS", raising=False)
    monkeypatch.delenv("AI_PRIOR_SIDE", raising=False)
    return tmp_path


def _row(direction="LONG", hit="1", resolved="2026-07-19T10:00:00", bar_ts=None,
        logged_at=None, pwin="", pwin_side="", fwd=""):
    return {"logged_at": logged_at if logged_at is not None else "2026-07-19T09:00:00",
            "symbol": "TST", "interval": "1h",
            "bar_ts": bar_ts if bar_ts is not None else "2026-07-19T09:00:00",
            "direction": direction, "action": direction,
            "resolved_at": resolved, "hit": hit, "pwin": pwin, "pwin_side": pwin_side,
            "fwd_ret_pct": fwd}


def _write_log(rows: list[dict]) -> None:
    with learning.PRED_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in FIELDS}, **r})


# =============================================================================
# 1) Varsayılan bit-denklik (AI_PRIOR_DECAY kurulu değilken)
# =============================================================================
def test_prior_from_state_decay_kapaliyken_bit_denk(monkeypatch):
    """`strategies_decayed` state'te DAHA İYİ bir prior taşısa bile, env kapalıyken tamamen
    yok sayılır — dönüş, yalnızca `strategies` varmış gibi hesaplanan değerle BİREBİR."""
    state = {
        "strategies": {"event_breakout": {"n": 50, "win": 40, "loss": 10,
                                          "post_mean": 0.79, "lo95": 0.66, "hi95": 0.89}},
        "strategies_decayed": {"event_breakout": {"n_ham": 500, "n": 480.0, "win": 470.0,
                                                   "loss": 10.0, "post_mean": 0.95,
                                                   "lo95": 0.93, "hi95": 0.97,
                                                   "half_life_days": 30.0,
                                                   "drift_shrink_uygulandi": False}},
    }
    monkeypatch.setattr(learning, "load_state", lambda: state)
    monkeypatch.delenv("AI_PRIOR_DECAY", raising=False)
    mean, lo, hi, src, decay_meta = decision._prior_from_state("event_breakout")
    assert (mean, lo, hi, src) == (0.79, 0.66, 0.89, "event_breakout")
    assert decay_meta is None


@pytest.mark.parametrize("kapali_deger", ["0", "false", "off", "no", "bozuk", ""])
def test_prior_from_state_acikca_kapali_degerlerde_de_bit_denk(monkeypatch, kapali_deger):
    state = {
        "strategies": {"GLOBAL": {"n": 100, "win": 60, "loss": 40,
                                  "post_mean": 0.6, "lo95": 0.51, "hi95": 0.69}},
        "strategies_decayed": {"GLOBAL": {"n_ham": 100, "n": 90.0, "win": 80.0, "loss": 10.0,
                                          "post_mean": 0.9, "lo95": 0.8, "hi95": 0.97,
                                          "half_life_days": 30.0,
                                          "drift_shrink_uygulandi": False}},
    }
    monkeypatch.setattr(learning, "load_state", lambda: state)
    monkeypatch.setenv("AI_PRIOR_DECAY", kapali_deger)
    mean, lo, hi, src, decay_meta = decision._prior_from_state("bilinmeyen_strateji")
    assert (mean, lo, hi, src) == (0.6, 0.51, 0.69, "GLOBAL")
    assert decay_meta is None


def test_decide_decay_kapaliyken_ucuca_bit_denk_ve_kanit_satiri_yok(monkeypatch):
    state = {
        "strategies": {"event_breakout": {"n": 50, "win": 40, "loss": 10,
                                          "post_mean": 0.79, "lo95": 0.66, "hi95": 0.89}},
        "strategies_decayed": {"event_breakout": {"n_ham": 500, "n": 480.0, "win": 470.0,
                                                   "loss": 10.0, "post_mean": 0.95,
                                                   "lo95": 0.93, "hi95": 0.97,
                                                   "half_life_days": 30.0,
                                                   "drift_shrink_uygulandi": False}},
    }
    monkeypatch.setattr(learning, "load_state", lambda: state)
    d = decision.decide("AAPL", "event_breakout", {})
    assert d.prior_mean == pytest.approx(0.79)
    assert not any("Zaman-ağırlıklı prior" in e for e in d.evidence)
    assert any("kaynak=event_breakout)" in e for e in d.evidence)


# =============================================================================
# 2) Yarılanma matematiği — kapalı formla (yaklaşık değil)
# =============================================================================
def test_age_weight_tam_yarilanma_omrunde_tam_yaridir():
    ref = datetime(2026, 8, 10, 12, 0, 0)
    half_life = learning.DECAY_HALF_LIFE_DAYS   # 30.0
    t_i = ref - timedelta(days=half_life)
    assert learning._age_weight(t_i, ref, half_life) == 0.5   # 0.5**1.0 — kapalı form, tam
    # çapraz-doğrulama: iki yarılanma ömrü → 0.25 (0.5**2)
    t_i2 = ref - timedelta(days=2 * half_life)
    assert learning._age_weight(t_i2, ref, half_life) == 0.25
    # yaş sıfır → ağırlık tam 1.0
    assert learning._age_weight(ref, ref, half_life) == 1.0


def test_age_weight_gelecek_damgasi_yasi_sifira_kirpilir():
    """t_i, ref'ten SONRA (negatif yaş) → 0'a kırpılır → ağırlık üst sınırı 1.0'ı aşmaz."""
    ref = datetime(2026, 8, 10)
    t_i = ref + timedelta(days=10)   # "gelecek" damga
    assert learning._age_weight(t_i, ref, 30.0) == 1.0


# =============================================================================
# 3) Ağırlıklı ≠ ağırlıksız — yön doğru (eski dilim kazanan / yeni dilim kaybeden)
# =============================================================================
def test_strategies_decayed_yon_dogru_eski_kazanan_yeni_kaybeden():
    eski, yeni = "2026-01-01T00:00:00", "2026-07-01T00:00:00"   # ~181 gün, half-life=30
    rows = [_row("LONG", "1", resolved=eski, bar_ts=eski) for _ in range(40)]
    rows += [_row("LONG", "0", resolved=yeni, bar_ts=yeni) for _ in range(40)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)

    # ağırlıksız: dengeli (40 kazanç/40 kayıp) → tam 0.5, A1'den ETKİLENMEDİ
    assert st["strategies"]["GLOBAL"]["n"] == 80
    assert st["strategies"]["GLOBAL"]["post_mean"] == pytest.approx(0.5)

    # ağırlıklı: yeni (tam ağırlıklı) kayıplar, eski (küçük ağırlıklı) kazançları ezer
    weighted = st["strategies_decayed"]["GLOBAL"]["post_mean"]
    assert weighted < st["strategies"]["GLOBAL"]["post_mean"]
    assert weighted < 0.3   # ölçülen ~0.038 (bkz. görev betiği doğrulaması)
    assert st["strategies_decayed"]["GLOBAL"]["n_ham"] == 80


# =============================================================================
# 4) Drift shrink → CI GENİŞLER, lo95 ASLA ARTMAZ (aynı sayımlarda)
# =============================================================================
def test_beta_posterior_drift_shrink_ci_genisler_lo95_dusar():
    """Saf matematik: AYNI (w, l) çiftini DRIFT_SHRINK'e bölmek CI'yı genişletir, lo95'i
    hiç artırmaz — "aynı sayımlarda" ifadesinin kilidi budur."""
    for w, l in [(80.0, 20.0), (50.0, 50.0), (5.0, 45.0)]:
        mean0, lo0, hi0 = learning.beta_posterior(w, l)
        mean1, lo1, hi1 = learning.beta_posterior(w / learning.DRIFT_SHRINK, l / learning.DRIFT_SHRINK)
        assert (hi1 - lo1) > (hi0 - lo0), f"genişlemedi: w={w} l={l}"
        assert lo1 <= lo0 + 1e-12, f"lo95 arttı: w={w} l={l}"
    # Sınır durumu (dokümantasyon amaçlı, ayrı assert): kanıt zaten YOKKEN (0,0) shrink'in
    # etkisi yoktur — zaten maksimum-belirsiz uniform prior'dayız, genişleyecek yer kalmamış.
    assert learning.beta_posterior(0.0, 0.0) == learning.beta_posterior(0.0 / learning.DRIFT_SHRINK,
                                                                         0.0 / learning.DRIFT_SHRINK)


def test_update_from_predictions_drift_shrink_bayragi_ve_etkin_n_yariya_iner():
    # drift YOK — tüm satırlar aynı bar_ts (yaş=0 → ağırlık=1.0, ham=etkin)
    rows = [_row("LONG", "1") for _ in range(60)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    assert st["drift"]["drift"] is False
    g = st["strategies_decayed"]["GLOBAL"]
    assert g["drift_shrink_uygulandi"] is False
    assert g["win"] == pytest.approx(60.0)
    assert g["loss"] == pytest.approx(0.0)

    # drift VAR (test_ai_learning.py::test_retrain_candidate_true_drift_tetiklenince deseni)
    rows2 = [_row("LONG", "1") for _ in range(70)] + [_row("LONG", "0") for _ in range(70)]
    _write_log(rows2)
    st2 = learning.update_from_predictions(tail_n=500)
    assert st2["drift"]["drift"] is True
    g2 = st2["strategies_decayed"]["GLOBAL"]
    assert g2["drift_shrink_uygulandi"] is True
    # ham toplam 140 (ağırlık=1.0 her satırda) → shrink sonrası etkin n YARIYA iner
    assert g2["n_ham"] == 140
    assert g2["n"] == pytest.approx(140.0 / learning.DRIFT_SHRINK)
    assert g2["win"] == pytest.approx(70.0 / learning.DRIFT_SHRINK)
    assert g2["loss"] == pytest.approx(70.0 / learning.DRIFT_SHRINK)


# =============================================================================
# 5) Referans-zaman determinizmi ("bugün" DEĞİL, kuyruktaki en son damga)
# =============================================================================
def test_referans_zaman_determinizmi_ayni_log_farkli_bugun(monkeypatch):
    rows = [_row("LONG", "1", resolved="2026-01-01T00:00:00", bar_ts="2026-01-01T00:00:00")
            for _ in range(20)]
    rows += [_row("LONG", "0", resolved="2026-06-01T00:00:00", bar_ts="2026-06-01T00:00:00")
            for _ in range(20)]
    _write_log(rows)

    monkeypatch.setattr(learning, "_utc_now_iso", lambda: "2020-01-01T00:00:00+00:00")
    st1 = learning.update_from_predictions(tail_n=500)

    monkeypatch.setattr(learning, "_utc_now_iso", lambda: "2099-01-01T00:00:00+00:00")
    st2 = learning.update_from_predictions(tail_n=500)

    assert st1["updated_at"] != st2["updated_at"]     # kanıt: "bugün" GERÇEKTEN farklıydı
    assert st1["strategies_decayed"] == st2["strategies_decayed"]
    assert st1["strategies"] == st2["strategies"]


# =============================================================================
# 6) tz kanonikleştirme — çökmez, karışık-tz karşılaştırma hatası vermez
# =============================================================================
def test_parse_ts_naive_utc_tz_aware_utcye_cevirir_ve_tzinfo_dusurur():
    dt = learning._parse_ts_naive_utc("2026-06-23 15:59:00-04:00")
    assert dt == datetime(2026, 6, 23, 19, 59, 0)
    assert dt.tzinfo is None


def test_parse_ts_naive_utc_naive_oldugu_gibi_kalir():
    assert learning._parse_ts_naive_utc("2026-07-31 19:54:00") == datetime(2026, 7, 31, 19, 54, 0)
    assert learning._parse_ts_naive_utc("2026-06-24T16:30:36") == datetime(2026, 6, 24, 16, 30, 36)


def test_parse_ts_naive_utc_bos_ve_bozuk_none():
    assert learning._parse_ts_naive_utc("") is None
    assert learning._parse_ts_naive_utc("tamamen-bozuk-zaman") is None


def test_parse_ts_naive_utc_cikti_daima_naive_karisik_tz_cikartma_hata_vermez():
    """Asıl kilit: biri aslen tz-aware biri aslen tz-naive iki damga, kanonikleştirmeden
    SONRA ikisi de naive'dir — çıkarma/karşılaştırma TypeError ATMAZ (BULGU-16 tekrarı)."""
    aslen_aware = learning._parse_ts_naive_utc("2026-06-23 15:59:00-04:00")
    aslen_naive = learning._parse_ts_naive_utc("2026-07-31 19:54:00")
    assert aslen_aware.tzinfo is None and aslen_naive.tzinfo is None
    delta = aslen_naive - aslen_aware   # TypeError atarsa test ÇÖKER (bu, kilidin kendisi)
    assert delta.total_seconds() > 0


def test_resolve_row_ts_oncelik_sirasi_ve_hicbiri_yoksa_none():
    dt = learning._resolve_row_ts("2026-05-01T00:00:00", "2026-06-01T00:00:00", "2026-07-01T00:00:00")
    assert dt == datetime(2026, 5, 1)                 # bar_ts öncelikli
    dt2 = learning._resolve_row_ts("", "2026-06-01T00:00:00", "2026-07-01T00:00:00")
    assert dt2 == datetime(2026, 6, 1)                # bar_ts yok → logged_at
    dt3 = learning._resolve_row_ts("", "", "2026-07-01T00:00:00")
    assert dt3 == datetime(2026, 7, 1)                # ikisi de yok → resolved_at
    assert learning._resolve_row_ts("", "", "") is None
    assert learning._resolve_row_ts("bozuk", "daha-bozuk", "yine-bozuk") is None


def test_karisik_tz_bicimleri_gercek_semada_cokmez():
    """Canlı log'daki GERÇEK karışıklık: kimi bar_ts'te tz-ofset var, kimi tz'siz/boşluklu,
    kimi tz'siz/T'li (2026-08-10'da data/predictions_log.csv'den doğrulandı)."""
    rows = [
        _row("LONG", "1", resolved="2026-06-24T16:55:34", bar_ts="2026-06-23 15:59:00-04:00"),
        _row("LONG", "1", resolved="2026-07-31T22:55:00", bar_ts="2026-07-31 19:54:00"),
        _row("LONG", "0", resolved="2026-07-15T10:00:00", bar_ts="2026-07-15T10:00:00"),
    ]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)   # çökmemeli
    assert st["n_unparsed"] == 0
    assert st["strategies_decayed"]["GLOBAL"]["n_ham"] == 3


# =============================================================================
# 7) Parse edilemeyen damga → n_unparsed artar, ana (ağırlıksız) sayıma girmez, çökme yok
# =============================================================================
def test_ayristirilamayan_damga_n_unparsed_artar_agirlikli_sayima_girmez():
    rows = [_row("LONG", "1", resolved="2026-07-19T10:00:00", bar_ts="2026-07-19T10:00:00")
            for _ in range(5)]
    # resolved_at BOŞ-OLMAYAN ama ISO-DIŞI (ilk filtreyi geçer), bar_ts/logged_at boş.
    rows += [_row("LONG", "1", resolved="COZULEMEZ_DAMGA", bar_ts="", logged_at="")
            for _ in range(3)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    assert st["n_unparsed"] == 3
    assert st["strategies"]["GLOBAL"]["n"] == 8          # ağırlıksız sayım ETKİLENMEDİ
    assert st["strategies_decayed"]["GLOBAL"]["n_ham"] == 5   # yalnız parse-edilebilenler


def test_hicbir_satir_parse_edilemezse_strategies_decayed_bos_cokmez():
    rows = [_row("LONG", "1", resolved="COZULEMEZ", bar_ts="", logged_at="") for _ in range(4)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)   # çökmemeli
    assert st["n_unparsed"] == 4
    assert st["strategies_decayed"] == {}
    assert st["strategies"]["GLOBAL"]["n"] == 4          # ağırlıksız yine sağlam


# =============================================================================
# 8) beta_posterior — mevcut TAM SAYI çağrılarının bit-denkliği
# =============================================================================
def test_beta_posterior_int_cagri_bit_denk():
    """test_ai_learning.py::test_beta_posterior_elle ile AYNI değerler (yeni float imza
    eski int-çağrı davranışını bit-bit korur)."""
    mean, lo, hi = learning.beta_posterior(2, 0)
    assert mean == pytest.approx(0.75)
    assert 0.0 < lo < mean < hi < 1.0
    mean0, lo0, hi0 = learning.beta_posterior(0, 0)
    assert mean0 == pytest.approx(0.5)
    assert hi0 - lo0 > 0.9
    assert learning.beta_posterior(-5, -5)[0] == pytest.approx(0.5)
    # asıl bit-denklik kanıtı: int ve float çağrı TAMAMEN aynı üçlüyü üretir
    assert learning.beta_posterior(7, 3) == learning.beta_posterior(7.0, 3.0)
    assert learning.beta_posterior(0, 0) == learning.beta_posterior(0.0, 0.0)


# =============================================================================
# 9) n_ham eşiği — yan-koşullu prior'ın _MIN_SIDE_PRIOR_N kapısı HAM sayıya bakar
# =============================================================================
def test_n_ham_esigi_agirlikli_kipte_ham_sayiya_bakar_etkin_sayima_degil(monkeypatch):
    """n_ham=250 (eşiği GEÇER) AMA etkin (float, decay'li) n yalnız 3.8 — eşik HAM sayıya
    baktığı için LONG kaynağı KABUL edilmeli (etkin sayıma bakılsaydı reddedilirdi)."""
    state = {
        "strategies": {"GLOBAL": {"n": 1000, "win": 500, "loss": 500,
                                  "post_mean": 0.5, "lo95": 0.47, "hi95": 0.53}},
        "strategies_decayed": {
            "LONG": {"n_ham": 250, "n": 3.8, "win": 3.5, "loss": 0.3,
                     "post_mean": 0.9, "lo95": 0.7, "hi95": 0.98,
                     "half_life_days": 30.0, "drift_shrink_uygulandi": False},
        },
    }
    monkeypatch.setenv("AI_PRIOR_DECAY", "1")
    monkeypatch.setattr(learning, "load_state", lambda: state)
    mean, lo, hi, src, decay_meta = decision._prior_from_state("baska_strateji", "LONG")
    assert src == "LONG~decay"
    assert mean == pytest.approx(0.9)
    assert decay_meta == {"half_life_days": 30.0, "drift_shrink_uygulandi": False}


def test_n_ham_esigi_yetersizse_bir_sonraki_kaynaga_duser(monkeypatch):
    state = {
        "strategies": {"GLOBAL": {"n": 1000, "win": 500, "loss": 500,
                                  "post_mean": 0.5, "lo95": 0.47, "hi95": 0.53}},
        "strategies_decayed": {
            "LONG": {"n_ham": 50, "n": 45.0, "win": 40.0, "loss": 5.0,     # n_ham < 200
                     "post_mean": 0.9, "lo95": 0.7, "hi95": 0.98,
                     "half_life_days": 30.0, "drift_shrink_uygulandi": False},
            "GLOBAL": {"n_ham": 1000, "n": 900.0, "win": 450.0, "loss": 450.0,
                       "post_mean": 0.5, "lo95": 0.47, "hi95": 0.53,
                       "half_life_days": 30.0, "drift_shrink_uygulandi": False},
        },
    }
    monkeypatch.setenv("AI_PRIOR_DECAY", "1")
    monkeypatch.setattr(learning, "load_state", lambda: state)
    mean, lo, hi, src, decay_meta = decision._prior_from_state("baska_strateji", "LONG")
    assert src == "GLOBAL~decay"    # LONG'un decayed n_ham'i yetersiz → GLOBAL'in decayed'i


def test_n_ham_esigi_ham_yeterliyken_kapali_ortamda_eskisiyle_ayni_esik_uygulanir(monkeypatch):
    """Kontrol: AI_PRIOR_DECAY KAPALIYKEN eşik ESKİ (ağırlıksız) `n` alanına bakar —
    davranış tamamen A1-öncesiyle aynıdır (regresyon değil)."""
    state = {
        "strategies": {
            "LONG": {"n": 150, "win": 100, "loss": 50,           # n < 200 → yetersiz
                     "post_mean": 0.67, "lo95": 0.6, "hi95": 0.74},
            "GLOBAL": {"n": 1000, "win": 500, "loss": 500,
                      "post_mean": 0.5, "lo95": 0.47, "hi95": 0.53},
        },
    }
    monkeypatch.delenv("AI_PRIOR_DECAY", raising=False)
    monkeypatch.setattr(learning, "load_state", lambda: state)
    mean, lo, hi, src, decay_meta = decision._prior_from_state("baska_strateji", "LONG")
    assert src == "GLOBAL"
    assert decay_meta is None


# =============================================================================
# Ekstra — AI_PRIOR_HALF_LIFE_DAYS: savunmacı env parse (geçersiz/NaN/≤0 → varsayılan)
# =============================================================================
def test_resolve_half_life_days_tanimsizken_varsayilan(monkeypatch):
    monkeypatch.delenv("AI_PRIOR_HALF_LIFE_DAYS", raising=False)
    assert learning._resolve_half_life_days() == learning.DECAY_HALF_LIFE_DAYS


def test_resolve_half_life_days_gecerli_deger_override(monkeypatch):
    monkeypatch.setenv("AI_PRIOR_HALF_LIFE_DAYS", "7")
    assert learning._resolve_half_life_days() == 7.0


@pytest.mark.parametrize("raw", ["", "bozuk-deger", "0", "-5", "nan"])
def test_resolve_half_life_days_savunmaci_varsayilana_duser(monkeypatch, raw):
    monkeypatch.setenv("AI_PRIOR_HALF_LIFE_DAYS", raw)
    assert learning._resolve_half_life_days() == learning.DECAY_HALF_LIFE_DAYS


def test_update_from_predictions_ortam_yarilanma_omrunu_degistirir(monkeypatch):
    rows = [_row("LONG", "1") for _ in range(10)]
    _write_log(rows)
    monkeypatch.setenv("AI_PRIOR_HALF_LIFE_DAYS", "7")
    st = learning.update_from_predictions(tail_n=500)
    assert st["strategies_decayed"]["GLOBAL"]["half_life_days"] == 7.0


# =============================================================================
# Ekstra — AI_PRIOR_DECAY env ayrıştırması (1/true/on/yes, büyük/küçük harf duyarsız)
# =============================================================================
@pytest.mark.parametrize("deger", ["1", "true", "TRUE", "True", "on", "On", "yes", " yes "])
def test_resolve_prior_decay_enabled_acik_degerler(monkeypatch, deger):
    monkeypatch.setenv("AI_PRIOR_DECAY", deger)
    assert decision._resolve_prior_decay_enabled() is True


@pytest.mark.parametrize("deger", ["0", "false", "off", "no", "", "bozuk", "2"])
def test_resolve_prior_decay_enabled_kapali_degerler(monkeypatch, deger):
    monkeypatch.setenv("AI_PRIOR_DECAY", deger)
    assert decision._resolve_prior_decay_enabled() is False


def test_resolve_prior_decay_enabled_tanimsizken_kapali(monkeypatch):
    monkeypatch.delenv("AI_PRIOR_DECAY", raising=False)
    assert decision._resolve_prior_decay_enabled() is False


# =============================================================================
# Uçtan-uca — A1 (learning) ürettiği gerçek strategies_decayed, A2 (decision) tarafından
# YALNIZ env açıkken tüketilir
# =============================================================================
def test_uctan_uca_gercek_log_decay_acikken_kararin_prior_kaynagini_degistirir(monkeypatch):
    eski, yeni = "2026-01-01T00:00:00", "2026-07-01T00:00:00"
    rows = [_row("LONG", "1", resolved=eski, bar_ts=eski) for _ in range(220)]
    rows += [_row("LONG", "0", resolved=yeni, bar_ts=yeni) for _ in range(220)]
    _write_log(rows)
    st = learning.update_from_predictions()   # varsayılan tail_n=5000, 440 satırı kapsar
    monkeypatch.setattr(learning, "load_state", lambda: st)

    monkeypatch.delenv("AI_PRIOR_DECAY", raising=False)
    d_off = decision.decide("AAPL", "event_breakout", {}, side_hint="LONG")

    monkeypatch.setenv("AI_PRIOR_DECAY", "1")
    d_on = decision.decide("AAPL", "event_breakout", {}, side_hint="LONG")

    assert any("kaynak=LONG)" in e for e in d_off.evidence)
    assert not any("Zaman-ağırlıklı prior" in e for e in d_off.evidence)

    assert any("kaynak=LONG~decay)" in e for e in d_on.evidence)
    assert any("Zaman-ağırlıklı prior" in e for e in d_on.evidence)

    assert d_off.prior_mean == pytest.approx(0.5, abs=1e-9)     # 220 kazanç/220 kayıp, dengeli
    assert d_on.prior_mean < 0.3                                # yeni kayıplar ağır basıyor
    assert d_on.prior_mean != pytest.approx(d_off.prior_mean)
