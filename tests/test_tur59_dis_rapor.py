"""
Tur-59 kilitleri — dış rapor triyajının koda geçen kısmı.

Her test bir BULGUyu kilitler; hiçbiri kenar iddiası taşımaz.
  BULGU-59-1  özellik logu politikası: aday-şartı KALKTI (etiketsiz havuz ~8×)
  BULGU-59-2  intra-bar sıra: açılış İKİ yönde de kanıttır + belirsizlik SAYILIR
  BULGU-59-3  hacim-katılımına duyarlı piyasa etkisi (varsayılan KAPALI → bit-denk)
  BULGU-59-4  `tier` kolonu: katman-başı defter ayrıştırması
  BULGU-59-5  trades.csv BAŞLIK GÖÇÜ yoktu (kolon eklenince satırlar kayardı)
  BULGU-59-6  beta-ağırlıklı yönsel maruziyet + endeks hedge ÖNERİSİ (ölçüm koşuluyla)
  BULGU-59-7  piyasa bağlamı kolonları (genişlik/göreceli güç/çapraz varlık)
  BULGU-59-8  float kaynağı bağlandı → `turnover_rate` ölçülebilir
"""
from __future__ import annotations

import csv
import gzip
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.config import CONFIG
from nihai.execution.dual_tier import TIER_NAMES, current_tier
from nihai.risk.position_manager import PositionManager, PositionTracker, SlippageModel
from nihai.risk.risk_manager import PortfolioState, RiskManager, VirtualPosition
from nihai.signals import breadth as BR
from nihai.signals.market_regime import beta_vs_benchmark


# ---------------------------------------------------------------------------
# yardımcılar
# ---------------------------------------------------------------------------
def _pos(side="LONG", stop=95.0, target=110.0, **kw) -> VirtualPosition:
    return VirtualPosition(symbol="AAA", asset_type="stock", side=side, quantity=10,
                           entry_price=100.0, entry_time=date(2026, 1, 2),
                           stop=stop, target=target, **kw)


def _mgr(pos: VirtualPosition) -> tuple[PositionManager, PositionTracker]:
    st = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    st.open_positions.append(pos)
    m = PositionManager(st, SlippageModel(), record_logs=False)
    t = PositionTracker(pos=pos, signal_reasons=[], sizing={}, current_price=pos.entry_price)
    t.initial_stop = pos.stop
    m._trackers[m._key(pos)] = t
    return m, t


NOW = datetime(2026, 1, 3, 15, 0)


# ===========================================================================
# BULGU-59-2 — intra-bar sıra
# ===========================================================================
def test_acilis_hedefin_otesindeyse_GAP_TARGET_yazilir():
    """LONG: bar hedefin ÜSTÜNDE açıldı → hedef ilk tikte dolu. Eskiden STOP yazılıyordu."""
    m, t = _mgr(_pos())
    r = m.check_exit(t, bar_high=115.0, bar_low=94.0, bar_open=112.0, bar_close=113.0,
                     now_dt=NOW)
    assert r is not None and r[0] == "GAP_TARGET"
    assert r[1] == 112.0            # dolum AÇILIŞTAN (hedeften daha iyi fiyat gerçekti)


def test_short_gap_target_simetrik():
    m, t = _mgr(_pos(side="SHORT", stop=105.0, target=90.0))
    r = m.check_exit(t, bar_high=106.0, bar_low=85.0, bar_open=88.0, bar_close=87.0,
                     now_dt=NOW)
    assert r is not None and r[0] == "GAP_TARGET" and r[1] == 88.0


def test_gap_stop_hala_hedeften_ONCE():
    """Açılış stop'un ALTINDAysa hedef gap'i sorulmaz — zarar-kesme önceliği korunur."""
    m, t = _mgr(_pos())
    r = m.check_exit(t, bar_high=120.0, bar_low=90.0, bar_open=93.0, bar_close=118.0,
                     now_dt=NOW)
    assert r is not None and r[0] == "GAP_STOP"


def test_stop_first_politikasi_eski_davranisi_geri_verir():
    """`INTRABAR_POLICY=stop_first` Tur-58 davranışı — A/B ve geri alma için.

    CONFIG dataclass'ları FROZEN; `cost_attribution` ile aynı desen (`object.__setattr__`
    + finally ile geri koyma) kullanılır."""
    bc = CONFIG.backtest
    eski = bc.intrabar_policy
    object.__setattr__(bc, "intrabar_policy", "stop_first")
    try:
        m, t = _mgr(_pos())
        r = m.check_exit(t, bar_high=115.0, bar_low=94.0, bar_open=112.0, bar_close=113.0,
                         now_dt=NOW)
        assert r is not None and r[0] == "STOP"
    finally:
        object.__setattr__(bc, "intrabar_policy", eski)


def test_gercek_belirsizlik_sayilir_ve_STOP_kotumser_kalir():
    """Açılış İKİSİNİN ARASINDA + ikisi de delindi → sıra bilinmez → stop-önce + sayaç."""
    m, t = _mgr(_pos())
    r = m.check_exit(t, bar_high=115.0, bar_low=94.0, bar_open=100.0, bar_close=108.0,
                     now_dt=NOW)
    assert r is not None and r[0] == "STOP"
    assert m.intrabar_ambiguous == 1
    assert m.intrabar_resolved == 0 and m.intrabar_flipped == 0


def test_belirsiz_olmayan_barda_sayac_ARTMAZ():
    m, t = _mgr(_pos())
    m.check_exit(t, bar_high=104.0, bar_low=94.0, bar_open=100.0, bar_close=96.0, now_dt=NOW)
    assert m.intrabar_ambiguous == 0


def test_resolver_sirayi_cevirebilir_ve_sayilir():
    m, t = _mgr(_pos())
    m.intrabar_resolver = lambda pos, bar_open, now: "TARGET"
    r = m.check_exit(t, bar_high=115.0, bar_low=94.0, bar_open=100.0, bar_close=108.0,
                     now_dt=NOW)
    assert r is not None and r[0] == "TARGET"
    assert (m.intrabar_ambiguous, m.intrabar_resolved, m.intrabar_flipped) == (1, 1, 1)


def test_resolver_None_donerse_kotumser_varsayima_DUSER():
    """Ölçüm yokluğu ölçüm gibi sunulmaz: veri yoksa stop-önce KALIR (Tur-31)."""
    m, t = _mgr(_pos())
    m.intrabar_resolver = lambda pos, bar_open, now: None
    r = m.check_exit(t, bar_high=115.0, bar_low=94.0, bar_open=100.0, bar_close=108.0,
                     now_dt=NOW)
    assert r is not None and r[0] == "STOP"
    assert m.intrabar_resolved == 0 and m.intrabar_flipped == 0


def test_resolver_patlarsa_akis_bozulmaz():
    def _patla(pos, bar_open, now):
        raise RuntimeError("veri yok")
    m, t = _mgr(_pos())
    m.intrabar_resolver = _patla
    r = m.check_exit(t, bar_high=115.0, bar_low=94.0, bar_open=100.0, bar_close=108.0,
                     now_dt=NOW)
    assert r is not None and r[0] == "STOP"


def test_backtest_sonucu_belirsizlik_sayaclarini_TASIR():
    """Yazılıp OKUNMAYAN teşhis bu projede kusurdur (Tur-38) — alan sonuca girmeli."""
    from nihai.execution.backtest import BacktestResult, _intrabar_stats
    r = BacktestResult("X", "1d", 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0)
    assert r.intrabar_ambiguous == 0
    st = PortfolioState(equity=1.0, ath_equity=1.0, cash_balance=1.0)
    m = PositionManager(st, SlippageModel(), record_logs=False)
    m.intrabar_ambiguous, m.intrabar_flipped = 7, 2
    assert _intrabar_stats(m) == {"intrabar_ambiguous": 7, "intrabar_resolved": 0,
                                  "intrabar_flipped": 2}
    assert _intrabar_stats(None)["intrabar_ambiguous"] == 0     # geriye uyum


# ===========================================================================
# BULGU-59-3 — piyasa etkisi
# ===========================================================================
def test_impact_varsayilan_KAPALI_bit_denk():
    s = SlippageModel()
    assert s.impact_coef == 0.0
    assert s.slip_pct("stock", qty=1e9, adv_shares=1.0, sigma=9.9) == s.stock_slippage_pct
    assert s.apply_entry(100.0, "LONG", "stock") == 100.0 * 1.001


def test_impact_karekok_yasasi_dogru_yonde():
    s = SlippageModel(impact_coef=1.0)
    base = s.stock_slippage_pct
    kucuk = s.slip_pct("stock", qty=100, adv_shares=1_000_000, sigma=0.02)
    buyuk = s.slip_pct("stock", qty=100_000, adv_shares=1_000_000, sigma=0.02)
    assert base < kucuk < buyuk
    # Kapalı form: taban + η·σ·sqrt(Q/ADV)
    assert buyuk == pytest.approx(base + 1.0 * 0.02 * (0.1 ** 0.5))


def test_impact_olcum_kosulu_eksikse_TABANA_duser():
    s = SlippageModel(impact_coef=1.0)
    b = s.stock_slippage_pct
    assert s.slip_pct("stock", qty=100, adv_shares=None, sigma=0.02) == b
    assert s.slip_pct("stock", qty=100, adv_shares=1e6, sigma=None) == b
    assert s.slip_pct("stock", qty=100, adv_shares=float("nan"), sigma=0.02) == b
    assert s.slip_pct("stock", qty=0, adv_shares=1e6, sigma=0.02) == b


def test_impact_cikista_da_KOTU_tarafa_kayar():
    s = SlippageModel(impact_coef=1.0)
    kw = {"qty": 100_000, "adv_shares": 1_000_000, "sigma": 0.02}
    assert s.apply_exit(100.0, "LONG", "stock", **kw) < 100.0 * (1 - s.stock_slippage_pct)
    assert s.apply_exit(100.0, "SHORT", "stock", **kw) > 100.0 * (1 + s.stock_slippage_pct)


def test_causal_adv_pandas_ile_ayni():
    from nihai.execution.backtest import _causal_adv
    v = np.array([5.0, 3, 8, 1, 9, 2, 7])
    beklenen = pd.Series(v).rolling(3, min_periods=1).mean().to_numpy()
    assert np.allclose(_causal_adv(v, 3), beklenen)


def test_causal_adv_gelecege_BAKMAZ():
    """i konumundaki değer YALNIZ [0..i] barlarından; i+1'i değiştirmek onu değiştirmez."""
    from nihai.execution.backtest import _causal_adv
    v = np.array([5.0, 3, 8, 1, 9])
    a = _causal_adv(v, 3)
    v2 = v.copy(); v2[4] = 1e9
    assert np.allclose(a[:4], _causal_adv(v2, 3)[:4])


def test_pozisyon_adv_sigmayi_TASIR_ve_cikisi_etkiler():
    st = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    m = PositionManager(st, SlippageModel(impact_coef=1.0), record_logs=False)
    p = _pos(adv_shares=1_000_000.0, bar_sigma=0.02)
    st.open_positions.append(p)
    t = PositionTracker(pos=p, signal_reasons=[], sizing={}, current_price=100.0)
    m._trackers[m._key(p)] = t
    ev = m._close(p, 100.0, "TEST", NOW)
    assert ev.exit_price < 100.0 * (1 - SlippageModel().stock_slippage_pct)


# ===========================================================================
# BULGU-59-4/5 — katman kolonu + trades.csv başlık göçü
# ===========================================================================
def test_current_tier_ana_threadde_BOS():
    assert current_tier() == ""


def test_current_tier_katman_threadinde_ADI_verir():
    import threading
    got = {}
    th = threading.Thread(target=lambda: got.setdefault("t", current_tier()),
                          name=TIER_NAMES[0])
    th.start(); th.join()
    assert got["t"] == TIER_NAMES[0]


def test_trades_csv_basligi_tier_TASIR():
    from nihai.signals.signal_engine import TRADES_CSV_HEADER
    assert TRADES_CSV_HEADER[-1] == "tier"


def test_trades_csv_ESKI_baslikli_dosyayi_GOC_ETTIRIR(monkeypatch, tmp_path):
    """BULGU-59-5: göç olmadan eski başlığın altına yeni alan sırası yazılır → KOLON KAYAR."""
    import nihai.signals.signal_engine as SE
    yol = tmp_path / "trades.csv"
    eski = [c for c in SE.TRADES_CSV_HEADER if c != "tier"]
    with yol.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eski)
        w.writeheader()
        w.writerow(dict.fromkeys(eski, "x") | {"symbol": "OLD", "sector": "TECH"})
    monkeypatch.setattr(SE, "TRADES_CSV", yol)
    SE._ensure_trades_header()
    with yol.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == SE.TRADES_CSV_HEADER
    assert rows[0]["symbol"] == "OLD" and rows[0]["sector"] == "TECH"   # kayma YOK
    assert rows[0]["tier"] == ""                                       # yeni alan boş


def test_trades_csv_yazimi_tier_alanini_doldurur(monkeypatch, tmp_path):
    import nihai.signals.signal_engine as SE
    yol = tmp_path / "trades.csv"
    monkeypatch.setattr(SE, "TRADES_CSV", yol)
    tr = SE.TradeRecord(asset_label="AAA", strategy="S", entry_time=datetime(2026, 1, 2),
                        entry_price=100.0, entry_greeks=None, entry_iv=None,
                        entry_iv_rank=None, dte_at_entry=None,
                        contracts_or_shares=1, risk_dollars=10.0,
                        planned_rr=2.0, entry_reasons=[], exit_time=datetime(2026, 1, 3),
                        exit_price=110.0, exit_trigger="TARGET", realized_pnl=10.0)
    SE.log_trade_csv(tr, _pos(tier="odak"))
    rows = list(csv.DictReader(yol.open(encoding="utf-8", newline="")))
    assert rows[0]["tier"] == "odak" and rows[0]["symbol"] == "AAA"


# ===========================================================================
# BULGU-59-6 — beta maruziyeti
# ===========================================================================
def _seri(n=200, beta=1.5, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    rb = rng.normal(0, 0.01, n)
    rs = beta * rb + rng.normal(0, 0.001, n)
    return (pd.Series(50 * np.exp(np.cumsum(rs)), index=idx),
            pd.Series(100 * np.exp(np.cumsum(rb)), index=idx))


def test_beta_bilinen_dgpyi_geri_verir():
    s, b = _seri(beta=1.5)
    assert beta_vs_benchmark(s, b, window=180) == pytest.approx(1.5, abs=0.05)


def test_beta_olculemezse_None():
    s, b = _seri()
    assert beta_vs_benchmark(s.head(5), b.head(5)) is None          # kısa
    duz = pd.Series([100.0] * len(b), index=b.index)
    assert beta_vs_benchmark(s, duz) is None                        # düz benchmark
    assert beta_vs_benchmark(None, b) is None


def test_beta_kesisimsiz_takvimde_None():
    s, b = _seri()
    b2 = b.copy()
    b2.index = b2.index + timedelta(days=5000)
    assert beta_vs_benchmark(s, b2) is None


def test_beta_maruziyeti_OLCULMEMISI_toplama_katmaz():
    st = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    st.open_positions = [_pos(beta=2.0), _pos()]        # ikincisinin betası ölçülmemiş
    e = st.beta_exposure()
    assert e["measured"] == 1 and e["unmeasured"] == 1 and e["coverage"] == 0.5
    assert e["net_beta_dollars"] == pytest.approx(2.0 * 1000.0)
    assert e["net_notional"] == pytest.approx(2000.0)   # notional İKİSİNİ de sayar


def test_hedge_onerisi_esik_altinda_None_ustunde_TERS_bacak():
    st = PortfolioState(equity=1000.0, ath_equity=1000.0, cash_balance=1000.0)
    st.open_positions = [_pos(beta=1.0)]                # 1000 $ nominal = 1.0× equity
    rm = RiskManager(st)
    assert rm.beta_hedge_suggestion() is None           # tavan 2.0× → eşik altı
    st.open_positions = [_pos(beta=1.0) for _ in range(3)]   # 3000 $ = 3.0×
    s = rm.beta_hedge_suggestion(bench_price=100.0)
    assert s is not None and s["side"] == "SHORT"
    assert s["notional"] == pytest.approx(1000.0) and s["shares"] == 10


def test_hedge_onerisi_olcum_kapsamini_TASIR():
    st = PortfolioState(equity=1000.0, ath_equity=1000.0, cash_balance=1000.0)
    st.open_positions = [_pos(beta=3.0), _pos()]
    s = RiskManager(st).beta_hedge_suggestion()
    assert s is not None and s["coverage"] == 0.5 and s["unmeasured"] == 1


def test_hedge_onerisi_hic_olcum_yoksa_None():
    st = PortfolioState(equity=1000.0, ath_equity=1000.0, cash_balance=1000.0)
    st.open_positions = [_pos()]
    assert RiskManager(st).beta_hedge_suggestion() is None


# ===========================================================================
# BULGU-59-1/7 — özellik logu politikası + piyasa bağlamı
# ===========================================================================
def test_feat_fields_cand_side_ve_baglam_kolonlarini_TASIR():
    from nihai.execution.prediction_tracker import FEAT_FIELDS
    from nihai.signals.meta_features import FEATURE_NAMES
    assert "cand_side" in FEAT_FIELDS
    for c in BR.CONTEXT_FIELDS:
        assert c in FEAT_FIELDS
        assert c not in FEATURE_NAMES        # model girdisi DEĞİŞMEDİ


def test_ozellik_logu_aday_DISI_satiri_da_yazar(monkeypatch, tmp_path):
    import nihai.execution.prediction_tracker as PT
    monkeypatch.setattr(PT, "FEATS_LOG", tmp_path / "feats.csv")
    feats = dict.fromkeys(PT.FEATURE_NAMES, 1.0)
    rows = [
        {"symbol": "AAA", "bar_ts": "2026-01-02 10:00",
         "_feats": {**feats, "_cand_side": "LONG", "_ctx": {"br_above_sma50": 0.5,
                                                            "br_n": 12}}},
        {"symbol": "BBB", "bar_ts": "2026-01-02 10:00",
         "_feats": {**feats, "_cand_side": "", "_ctx": {"br_above_sma50": None,
                                                        "br_n": 12}}},
    ]
    assert PT._write_features(rows, "5m", "2026-01-02T10:00:00") == 2
    got = list(csv.DictReader((tmp_path / "feats.csv").open(encoding="utf-8", newline="")))
    assert [r["cand_side"] for r in got] == ["LONG", ""]
    assert got[0]["br_above_sma50"] == "0.5"
    assert got[1]["br_above_sma50"] == ""        # ölçüm yok → BOŞ, 0 DEĞİL
    assert got[0]["br_n"] == "12"


def test_servis_ozellik_kapisi_ADAY_sartina_bagli_DEGIL():
    """Kaynak denetimi: kapı `side is not None or _FEATURE_LOG_ALL` olmalı (BULGU-59-1)."""
    import inspect

    import nihai.execution.live_predict_service as S
    src = inspect.getsource(S.PredictionService._bar_context)
    assert "side is not None or _FEATURE_LOG_ALL" in src
    assert S._FEATURE_LOG_ALL is True         # varsayılan AÇIK


def test_baglam_kolonlari_bir_ONCEKI_turdan_gelir():
    """`snapshot()` yalnız `roll()` sonrası değişir → tur içinde SABİT, karar barından ÖNCE."""
    mc = BR.MarketContext()
    up = pd.Series(np.linspace(100, 140, 80))
    for i in range(12):
        mc.observe(f"S{i}", up)
    assert mc.snapshot()["br_n"] == 0          # henüz roll YOK → önceki tur boş
    mc.roll()
    assert mc.snapshot()["br_n"] == 12


def test_pending_bar_basina_TEK_roll_saglar():
    mc = BR.MarketContext()
    assert mc.pending() == 0                   # gözlem düşmediyse roll ÇAĞRILMAMALI
    mc.observe("S", pd.Series(np.linspace(100, 140, 80)))
    assert mc.pending() == 1
    mc.roll()
    assert mc.pending() == 0


def test_breadth_demo_kendini_dogrular():
    BR.demo()                                   # assert tabanlı öz-denetim


# ===========================================================================
# BULGU-59-8 — float kaynağı / turnover
# ===========================================================================
def test_float_shares_bos_sembolde_None():
    from nihai.data.market_structure import float_shares
    assert float_shares("") is None


def test_float_shares_onbellegi_agsiz_okur(monkeypatch, tmp_path):
    import json

    import nihai.data.market_structure as MS
    yol = tmp_path / "float.json"
    yol.write_text(json.dumps({"AAA": {"shares": 1e9, "source": "float",
                                       "ts": 9e18, "asof": "2026-01-01"}}),
                   encoding="utf-8")
    monkeypatch.setattr(MS, "_FLOAT_CACHE", yol)
    r = MS.float_shares("AAA")
    assert r is not None and r["shares"] == 1e9 and r["source"] == "float"


def test_float_shares_outstanding_ACIKCA_istenmeden_donmez(monkeypatch, tmp_path):
    """`float` ile `outstanding` FARKLI büyüklüklerdir — sessizce ikame edilmez."""
    import json

    import nihai.data.market_structure as MS
    yol = tmp_path / "float.json"
    yol.write_text(json.dumps({"BBB": {"shares": 5e8, "source": "outstanding",
                                       "ts": 9e18, "asof": "2026-01-01"}}),
                   encoding="utf-8")
    monkeypatch.setattr(MS, "_FLOAT_CACHE", yol)
    assert MS.float_shares("BBB") is None
    assert MS.float_shares("BBB", allow_outstanding=True)["source"] == "outstanding"


def test_turnover_hucreleri_olcum_yoksa_NEDEN_tasir(monkeypatch):
    import nihai.web.api_quant as Q
    monkeypatch.setattr("nihai.data.market_structure.float_shares", lambda s, **k: None)
    c = Q._turnover_cells("AAA", np.array([1000.0]))
    assert c["turnover"] is None and c["turnover_neden"]


def test_turnover_olculunce_oran_ve_KAYNAK_birlikte_doner(monkeypatch):
    import nihai.web.api_quant as Q
    monkeypatch.setattr("nihai.data.market_structure.float_shares",
                        lambda s, **k: {"shares": 1e6, "source": "float",
                                        "asof": "2026-08-30"})
    c = Q._turnover_cells("AAA", np.array([10_000.0]))
    assert c["turnover"]["rate"] == pytest.approx(0.01)
    assert c["turnover"]["source"] == "float" and c["turnover_neden"] is None


# ===========================================================================
# Opsiyon zinciri arşivi (dış rapor A2)
# ===========================================================================
def test_opsiyon_arsivi_gun_ici_KESIT_ekler_ve_geri_okunur(tmp_path):
    from nihai.data import option_archive as OA
    calls = pd.DataFrame({"strike": [100.0], "bid": [2.0], "ask": [2.2],
                          "impliedVolatility": [0.3], "openInterest": [500],
                          "volume": [10]})
    ch = {"expiry": "2026-09-18", "calls": calls, "puts": None}
    rows = OA.chain_rows("SPY", ch, asof="2026-08-30T12:00:00", spot=640.0)
    assert len(rows) == 1 and rows[0]["cp"] == "C" and rows[0]["open_interest"] == "500"
    OA.write_rows(rows, out_dir=tmp_path)
    OA.write_rows(rows, out_dir=tmp_path)
    back = OA.read_day(date.today(), out_dir=tmp_path)
    assert len(back) == 2                     # gzip ÇOKLU ÜYE append okunabiliyor
    p = tmp_path / f"option_chain_{date.today().isoformat()}.csv.gz"
    with gzip.open(p, "rt", encoding="utf-8") as f:
        assert f.readline().startswith("asof,")


def test_opsiyon_arsivi_olculemeyeni_BOS_birakir(tmp_path):
    from nihai.data import option_archive as OA
    puts = pd.DataFrame({"strike": [100.0], "bid": [1.5], "ask": [np.nan],
                         "openInterest": [np.nan], "volume": [0]})
    rows = OA.chain_rows("SPY", {"expiry": "2026-09-18", "puts": puts}, asof="t")
    assert rows[0]["ask"] == "" and rows[0]["open_interest"] == ""
    assert rows[0]["volume"] == "0"           # 0 GERÇEK bir sayımdır, boş değil


def test_opsiyon_arsivi_bos_zincirde_ISTISNA_atmaz(tmp_path):
    from nihai.data import option_archive as OA
    assert OA.chain_rows("X", {}, asof="t") == []
    assert OA.write_rows([], out_dir=tmp_path) is None
    assert OA.read_day("2000-01-01", out_dir=tmp_path) == []


# ===========================================================================
# Ölü adam anahtarı (dış rapor G2)
# ===========================================================================
def test_deadman_seans_kapaliyken_ALARM_vermez(monkeypatch, tmp_path):
    from nihai.notifier import deadman as DM
    monkeypatch.setattr(DM, "_market_open", lambda: False)
    (tmp_path / "predictions_log.csv").write_text("x", encoding="utf-8")
    res = DM.check(data_dir=tmp_path)
    assert res["verdict"] == "SEANS_KAPALI" and res["stale"] == []


def test_deadman_takvim_okunamazsa_IDDIA_ETMEZ(monkeypatch, tmp_path):
    from nihai.notifier import deadman as DM
    monkeypatch.setattr(DM, "_market_open", lambda: None)
    assert DM.check(data_dir=tmp_path)["verdict"] == "BILINMIYOR"


def test_deadman_dosya_YOKLUGU_bayatlik_DEGIL(monkeypatch, tmp_path):
    """Hiç koşmamış bir kurulumu 'öldü' diye raporlamak yanlış alarmdır (Tur-31)."""
    from nihai.notifier import deadman as DM
    monkeypatch.setattr(DM, "_market_open", lambda: True)
    res = DM.check(data_dir=tmp_path)
    assert res["verdict"] == "SAGLIKLI"
    assert all(d["stale"] is None for d in res["files"].values())


def test_deadman_bayat_dosyayi_seans_acikken_YAKALAR(monkeypatch, tmp_path):
    import os as _os

    from nihai.notifier import deadman as DM
    monkeypatch.setattr(DM, "_market_open", lambda: True)
    p = tmp_path / "predictions_log.csv"
    p.write_text("x", encoding="utf-8")
    eski = datetime.now().timestamp() - 3600 * 5
    _os.utime(p, (eski, eski))
    res = DM.check(data_dir=tmp_path)
    assert res["verdict"] == "BAYAT" and "predictions_log" in res["stale"]
    assert "BAYAT" in DM.format_report(res)


def test_deadman_grace_toleransi_uygulanir(monkeypatch, tmp_path):
    import os as _os

    from nihai.notifier import deadman as DM
    monkeypatch.setattr(DM, "_market_open", lambda: True)
    p = tmp_path / "state.json"
    p.write_text("{}", encoding="utf-8")
    eski = datetime.now().timestamp() - 60 * 35      # sınır 30 dk
    _os.utime(p, (eski, eski))
    assert DM.check(data_dir=tmp_path)["verdict"] == "BAYAT"
    assert DM.check(grace_min=15, data_dir=tmp_path)["verdict"] == "SAGLIKLI"


# ===========================================================================
# Konfigürasyon alanlarının OKUYANI var (Tur-50 kuralı)
# ===========================================================================
def test_yeni_config_alanlarinin_OKUYANI_var():
    kok = Path(__file__).resolve().parents[1]
    okunan = "\n".join((kok / p).read_text(encoding="utf-8")
                       for p in ("nihai/risk/position_manager.py",
                                 "nihai/execution/backtest.py"))
    assert "intrabar_policy" in okunan
    assert "impact_coef" in okunan
    assert "impact_adv_bars" in okunan
