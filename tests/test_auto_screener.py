"""Oto-tarayıcı testleri — filtre, sıralama, HEPSİ/top-N/iki-katman modları."""
import types

import pytest

import nihai.execution.auto_screener as a


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """watchlist'i izole et (tmp dosya, sıfır state, sabit tohum)."""
    import nihai.execution.watchlist as wl
    monkeypatch.setattr(wl, "WATCHLIST_FILE", tmp_path / "wl.json")
    monkeypatch.setattr(wl, "WATCHLIST", ["AAPL"])
    monkeypatch.setattr(wl, "_state", None)
    return a


def _fake_result(symbol, *, trades, exp, sharpe=1.0, ruin=None):
    """backtest_symbol'ün döndürdüğü BacktestResult'ı taklit eden hafif nesne."""
    return types.SimpleNamespace(
        symbol=symbol, trades_count=trades, bars_total=200,
        win_rate=0.5, expectancy=exp, total_return_pct=exp,
        max_drawdown_pct=2.0, sharpe_annualized=sharpe,
        monte_carlo={"ruin_probability": ruin} if ruin is not None else None,
    )


# --- Filtre mantığı ---
def test_passes_filters_min_trades(iso, monkeypatch):
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_TRADES", 5)
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_EXPECTANCY", 0)
    monkeypatch.setattr(a, "AUTO_SCREEN_MAX_RUIN", 0)
    assert a._passes_filters({"trades": 4, "expectancy": 10, "ruin_prob": None}) is False
    assert a._passes_filters({"trades": 5, "expectancy": 10, "ruin_prob": None}) is True


def test_passes_filters_expectancy_and_ruin(iso, monkeypatch):
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_TRADES", 1)
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_EXPECTANCY", 0)
    monkeypatch.setattr(a, "AUTO_SCREEN_MAX_RUIN", 0.10)
    assert a._passes_filters({"trades": 9, "expectancy": -1, "ruin_prob": None}) is False  # negatif exp
    assert a._passes_filters({"trades": 9, "expectancy": 5, "ruin_prob": 0.20}) is False   # ruin yüksek
    assert a._passes_filters({"trades": 9, "expectancy": 5, "ruin_prob": 0.05}) is True
    assert a._passes_filters({"trades": 9, "expectancy": 5, "ruin_prob": None}) is True     # ruin yok → geçer


# --- scan_universe: sıralama + filtre ---
def test_scan_universe_sorts_and_filters(iso, monkeypatch):
    monkeypatch.setattr(a, "build_universe", lambda name: ["LOW", "HIGH", "MID", "FEW"])
    # Seri yolu zorla: lambda mock process havuzuna (spawn) taşınamaz/pickle olmaz →
    # workers>1 olsaydı child process'ler GERÇEK backtest_symbol'ü çağırıp ağ'a giderdi.
    monkeypatch.setattr(a, "AUTO_SCREEN_WORKERS", 1)
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_TRADES", 5)
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_EXPECTANCY", 0)
    monkeypatch.setattr(a, "AUTO_SCREEN_MAX_RUIN", 0)
    monkeypatch.setattr(a, "AUTO_SCREEN_SORT", "expectancy")
    monkeypatch.setattr(a, "AUTO_SCREEN_THROTTLE", 0)
    fakes = {
        "LOW": _fake_result("LOW", trades=10, exp=5),
        "HIGH": _fake_result("HIGH", trades=10, exp=99),
        "MID": _fake_result("MID", trades=10, exp=40),
        "FEW": _fake_result("FEW", trades=2, exp=99),   # az işlem → elenmeli
    }
    monkeypatch.setattr(a, "backtest_symbol", lambda sym, **k: fakes[sym])
    ranked = a.scan_universe()
    syms = [r["symbol"] for r in ranked]
    assert syms == ["HIGH", "MID", "LOW"]   # expectancy'ye göre sıralı, FEW elendi


# --- HEPSİ modu (dual kapalı, TOP_N<=0): backtest atlanır ---
def test_hepsi_mode_skips_backtest(iso, monkeypatch):
    monkeypatch.setattr(a, "DUAL_TIER_ENABLED", False)
    monkeypatch.setattr(a, "AUTO_SCREEN_TOP_N", 0)
    monkeypatch.setattr(a, "build_universe", lambda name: ["X", "Y", "Z"])
    def _boom(**k):
        raise AssertionError("HEPSİ modunda backtest çağrılmamalı")
    monkeypatch.setattr(a, "scan_universe", _boom)
    new = a.scan_and_update()
    assert new == ["X", "Y", "Z"]


# --- Top-N modu (dual kapalı, TOP_N>0) ---
def test_topn_mode(iso, monkeypatch):
    monkeypatch.setattr(a, "DUAL_TIER_ENABLED", False)
    monkeypatch.setattr(a, "AUTO_SCREEN_TOP_N", 2)
    monkeypatch.setattr(a, "AUTO_SCREEN_SORT", "expectancy")
    monkeypatch.setattr(a, "scan_universe", lambda **k: [
        {"symbol": "A", "expectancy": 9, "ruin_prob": None},
        {"symbol": "B", "expectancy": 7, "ruin_prob": None},
        {"symbol": "C", "expectancy": 3, "ruin_prob": None}])
    new = a.scan_and_update()
    assert new == ["A", "B"]   # en iyi 2


# --- İki-katman: odak (en iyi N) + geniş (kalan), örtüşmesiz, tam kapsam ---
def test_dual_tier_partition(iso, monkeypatch):
    monkeypatch.setattr(a, "DUAL_TIER_ENABLED", True)
    monkeypatch.setattr(a, "FOCUSED_TOP_N", 2)
    monkeypatch.setattr(a, "AUTO_SCREEN_SORT", "expectancy")
    universe = ["A", "B", "C", "D", "E"]
    monkeypatch.setattr(a, "build_universe", lambda name: universe)
    monkeypatch.setattr(a, "scan_universe", lambda **k: [
        {"symbol": "C", "expectancy": 50, "ruin_prob": None},
        {"symbol": "A", "expectancy": 30, "ruin_prob": None},
        {"symbol": "E", "expectancy": 10, "ruin_prob": None}])
    new = a.scan_and_update()
    focused, rest = new[:2], new[2:]
    assert focused == ["C", "A"]                  # en iyi 2 sıralı (odak/TV)
    assert set(focused) & set(rest) == set()      # örtüşme yok
    assert set(new) == set(universe)              # tüm evren kapsandı
    assert a.watchlist.partition(20) == {"focused": focused, "broad": rest}
    monkeypatch.setattr(a, "scan_universe", lambda **k: [])
    assert a.scan_and_update() == new
    assert a.watchlist.partition(20)["focused"] == focused


def test_ranking_csv_written(iso, monkeypatch, tmp_path):
    monkeypatch.setattr(a, "AUTO_SCREEN_WRITE_CSV", True)
    monkeypatch.setattr(a, "LOG_DIR", tmp_path)
    rows = [{"symbol": "A", "expectancy": 9}, {"symbol": "B", "expectancy": 3}]
    path = a._write_ranking_csv(rows)
    assert path is not None
    content = open(path, encoding="utf-8").read()
    assert "symbol" in content and "A" in content and "B" in content
    assert len(list(tmp_path.glob("auto_screen_*.csv"))) == 1


def test_ranking_csv_disabled(iso, monkeypatch, tmp_path):
    monkeypatch.setattr(a, "AUTO_SCREEN_WRITE_CSV", False)
    monkeypatch.setattr(a, "LOG_DIR", tmp_path)
    assert a._write_ranking_csv([{"symbol": "A"}]) is None
    assert list(tmp_path.glob("auto_screen_*.csv")) == []


def test_bos_sp500_evreni_SESSIZ_kalmaz(monkeypatch):
    """S&P 500 boş dönerse SESSİZ kalmamalı — ve kapı ÇAĞIRANDA değil KAYNAKTA olmalı.

    ASİMETRİ: nasdaq100_tickers() çekemezse bayat yedek listeye düşer, sp500_tickers()
    ise BOŞ liste döndürür. 'both'ta union eskiden bunu yutuyordu — 7/24 tarayıcı ~600
    yerine 101 bayat isim tarar ve hiçbir kayıt "evren yarılandı" demezdi.

    Kapı `screen.sp500_tickers`ın İÇİNDE sınanır, `build_universe` üzerinden DEĞİL:
    aynı boşluğu DÖRT çağıran paylaşır (build_universe, screen.load_symbols,
    research/build_meta_dataset, research/xsect) ve yalnız birini sınayan bir sürüm
    diğer üçünü açıkta bırakırdı. Buradaki sahte tablo ayrıca 200 dönüp tablo parse
    EDİLEMEYEN dalı denetler — o dal hiçbir şey loglamıyordu. Tipik kök neden iletişim
    bilgisi taşımayan UA'dır (Wikimedia 403, bkz. nihai.net.contact_ua).
    """
    import pandas as pd
    from loguru import logger

    import nihai.execution.screen as s
    monkeypatch.setattr(s, "_read_html_tables", lambda url: [pd.DataFrame({"Symbol": []})])
    hatalar: list[str] = []
    sink = logger.add(hatalar.append, level="ERROR", format="{message}")
    try:
        semboller = s.sp500_tickers()
    finally:
        logger.remove(sink)
    assert semboller == []
    assert len(hatalar) == 1, f"boş S&P 500 için 1 ERROR bekleniyordu, {len(hatalar)} geldi"
    assert "S&P 500" in hatalar[0]


def test_dolu_sp500_evreninde_hata_kaydi_YOK(monkeypatch):
    """Karşı kutup: normal turda bu ERROR çıkmamalı (kapı kendini tatmin etmesin).
    Aynı koşuda `build_universe('both')` union'ının bozulmadığı da doğrulanır."""
    import pandas as pd
    from loguru import logger

    import nihai.execution.screen as s
    semboller = [f"S{i:03d}" for i in range(55)]   # _extract_ticker_col ≥50 satır ister
    monkeypatch.setattr(s, "_read_html_tables",
                        lambda url: [pd.DataFrame({"Symbol": semboller})])
    monkeypatch.setattr(a, "_UNIVERSE_MEMO", {})
    hatalar: list[str] = []
    sink = logger.add(hatalar.append, level="ERROR", format="{message}")
    try:
        evren = a.build_universe("both")
    finally:
        logger.remove(sink)
    assert evren == semboller
    assert hatalar == []
