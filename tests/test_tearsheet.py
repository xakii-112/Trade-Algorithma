"""P0-2 — nihai/web/tearsheet.py testleri (build_daily_returns + generate_tearsheet + uç).

Kaynak seçimi (data/trades.csv) ve equity-tabanı gerekçesi tearsheet.py modül
docstring'inde; burada o sözleşmeler kilitlenir:

- nlv_after TÜM satırlarda dolu → equity işaretleri DOĞRUDAN nlv_after (gün sonu = günün
  SON kapanışı); ilk getiri prior_equity = marks[0] − pnl[0] tabanına göre.
- Herhangi BİR satırda nlv_after eksik → TÜM seri INITIAL_EQUITY + cumsum(realized_pnl)
  (ya-hep-ya-hiç; karışık kaynak süreksizliği yasak).
- İşlem kapanmayan günler seride YOK (yapay 0-getiri satırı eklenmez).
- Boş/eksik dosya → boş Series (exception değil); bozuk satır sessizce elenir.
- generate_tearsheet: yetersiz veri (işlem<min VEYA gün<2) → quantstats'a dokunmadan
  Türkçe bilgi HTML'i; yeterli veri → GERÇEK quantstats HTML (pandas 2.x + matplotlib Agg
  entegrasyonu burada canlı doğrulanır) + atomik yazım (.tmp artığı kalmaz).
- /api/tearsheet: taze dosya varken yeniden üretim YOK; ?refresh=1 zorlar (/api/var
  deseninde sync def — burada yalnız davranış sözleşmesi test edilir).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import nihai.web.tearsheet as ts
import nihai.web.web_dashboard as wd

client = TestClient(wd.app)


def _df(rows: list[tuple[str, float, float | None]]) -> pd.DataFrame:
    """(exit_time, realized_pnl, nlv_after) üçlülerinden gerçek şemaya benzer DataFrame.
    Fazladan kolonlar kasıtlı: build_daily_returns yalnız gereken üçünü okumalı."""
    return pd.DataFrame({
        "exit_time": [r[0] for r in rows],
        "symbol": ["TST"] * len(rows),
        "realized_pnl": [r[1] for r in rows],
        "nlv_after": [r[2] for r in rows],
        "exit_trigger": ["target"] * len(rows),
    })


# ------------------------------------------------------------ build_daily_returns
def test_nlv_yolu_gun_sonu_isaretleri_ve_prior_equity():
    # 1. gün iki kapanış (gün sonu = SON kapanış 25050), 2. gün tek kapanış.
    df = _df([
        ("2026-06-01 10:30", +100.0, 25100.0),
        ("2026-06-01 15:45", -50.0, 25050.0),
        ("2026-06-03 11:00", +200.0, 25250.0),
    ])
    r = ts.build_daily_returns(df)
    assert len(r) == 2                                   # işlemsiz 2 Haziran seride YOK
    assert list(r.index) == [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-03")]
    prior = 25100.0 - 100.0                              # marks[0] − pnl[0] = 25000
    assert r.iloc[0] == pytest.approx(25050.0 / prior - 1.0)
    assert r.iloc[1] == pytest.approx(25250.0 / 25050.0 - 1.0)


def test_nlv_eksikse_tum_seri_cumsum_fallback():
    # TEK satırda nlv eksik → nlv'li satırlar dahil TÜM seri INITIAL_EQUITY+cumsum'a döner.
    df = _df([
        ("2026-06-01 10:30", +100.0, 25100.0),
        ("2026-06-02 11:00", -40.0, None),
        ("2026-06-04 12:00", +60.0, 25120.0),
    ])
    r = ts.build_daily_returns(df)
    base = ts.INITIAL_EQUITY
    marks = [base + 100.0, base + 60.0, base + 120.0]
    assert len(r) == 3
    assert r.iloc[0] == pytest.approx(marks[0] / base - 1.0)
    assert r.iloc[1] == pytest.approx(marks[1] / marks[0] - 1.0)
    assert r.iloc[2] == pytest.approx(marks[2] / marks[1] - 1.0)


def test_sirasiz_ve_bozuk_satirlar_temizlenir():
    df = _df([
        ("2026-06-05 10:00", +50.0, 25050.0),
        ("BOZUK-TARIH", +999.0, 25999.0),                # exit_time çözülemez → elenir
        ("2026-06-01 10:00", -30.0, 24970.0),
    ])
    # Kolon sırası _df ile aynı: exit_time, symbol, realized_pnl, nlv_after, exit_trigger
    df.loc[3] = ["2026-06-02 10:00", "TST", "sayı-değil", 25000.0, "stop"]  # pnl bozuk → elenir
    r = ts.build_daily_returns(df)
    assert len(r) == 2                                   # yalnız 06-01 ve 06-05 kaldı
    assert list(r.index) == [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-05")]


def test_bos_ve_eksik_dosya_bos_seri(tmp_path):
    assert ts.build_daily_returns(_df([])).empty
    assert ts.build_daily_returns(tmp_path / "yok.csv").empty


def test_csv_yolu_ile_okuma(tmp_path):
    p = tmp_path / "trades.csv"
    _df([("2026-06-01 10:00", +10.0, 25010.0),
         ("2026-06-02 10:00", -5.0, 25005.0)]).to_csv(p, index=False)
    r = ts.build_daily_returns(p)
    assert len(r) == 2


# ------------------------------------------------------------ generate_tearsheet
def test_yetersiz_islem_bilgi_html(tmp_path, monkeypatch):
    csv = tmp_path / "trades.csv"
    _df([("2026-06-01 10:00", +10.0, 25010.0),
         ("2026-06-02 10:00", -5.0, 25005.0),
         ("2026-06-03 10:00", +7.0, 25012.0)]).to_csv(csv, index=False)
    monkeypatch.setattr(ts, "TRADES_CSV", csv)
    out = tmp_path / "tearsheet.html"
    ts.generate_tearsheet(out, min_trades=10)
    html = out.read_text(encoding="utf-8")
    assert "Tear-Sheet henüz üretilemiyor" in html
    assert ">3</span>" in html                           # kapanan işlem sayısı gösterilir
    assert not (tmp_path / "tearsheet.html.tmp").exists()


def test_tek_gune_sikismis_islemler_de_yetersiz(tmp_path, monkeypatch):
    # 12 işlem ≥ min_trades AMA hepsi aynı gün → n_days=1 < 2 → yine bilgi HTML'i.
    rows = [(f"2026-06-01 {h:02d}:00", 5.0, 25000.0 + 5.0 * (i + 1))
            for i, h in enumerate(range(9, 21))]
    csv = tmp_path / "trades.csv"
    _df(rows).to_csv(csv, index=False)
    monkeypatch.setattr(ts, "TRADES_CSV", csv)
    out = tmp_path / "t.html"
    ts.generate_tearsheet(out, min_trades=10)
    assert "Tear-Sheet henüz üretilemiyor" in out.read_text(encoding="utf-8")


def test_yeterli_veri_gercek_quantstats_html(tmp_path, monkeypatch):
    pytest.importorskip("quantstats", reason="quantstats kurulu değil")
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2026-04-01", periods=40)
    nlv = 25_000.0
    rows = []
    for d, pnl in zip(days, rng.normal(20.0, 120.0, len(days)), strict=True):
        nlv += float(pnl)
        rows.append((f"{d.date()} 16:30", float(pnl), nlv))
    csv = tmp_path / "trades.csv"
    _df(rows).to_csv(csv, index=False)
    monkeypatch.setattr(ts, "TRADES_CSV", csv)

    out = tmp_path / "tearsheet.html"
    result = ts.generate_tearsheet(out, min_trades=10)
    assert result == out and out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "Sharpe" in html                              # quantstats metrik tablosu
    assert len(html) > 10_000                            # gerçek tear-sheet, bilgi sayfası değil
    assert not (tmp_path / "tearsheet.html.tmp").exists()   # atomik yazım artığı yok


# ------------------------------------------------------------ /api/tearsheet ucu
def test_api_taze_dosya_yeniden_uretim_yok(tmp_path, monkeypatch):
    html = tmp_path / "tearsheet.html"
    html.write_text("<html>TAZE-ICERIK</html>", encoding="utf-8")
    monkeypatch.setattr(ts, "TEARSHEET_HTML", html)

    def patlat(*a, **k):
        raise AssertionError("taze dosya varken generate_tearsheet çağrılmamalı")

    monkeypatch.setattr(ts, "generate_tearsheet", patlat)
    resp = client.get("/api/tearsheet")
    assert resp.status_code == 200
    assert "TAZE-ICERIK" in resp.text


def test_api_refresh_yeniden_uretimi_zorlar(tmp_path, monkeypatch):
    html = tmp_path / "tearsheet.html"
    html.write_text("<html>ESKI</html>", encoding="utf-8")
    monkeypatch.setattr(ts, "TEARSHEET_HTML", html)
    calls = {"n": 0}

    def sahte_uret(path, min_trades=10):
        calls["n"] += 1
        Path(path).write_text("<html>YENI</html>", encoding="utf-8")
        return Path(path)

    monkeypatch.setattr(ts, "generate_tearsheet", sahte_uret)
    resp = client.get("/api/tearsheet", params={"refresh": 1})
    assert resp.status_code == 200
    assert calls["n"] == 1
    assert "YENI" in resp.text
