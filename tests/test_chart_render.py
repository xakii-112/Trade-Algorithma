"""B2 — Telegram /grafik testleri: PNG render (sahte provider) + komut yönlendirme."""
import numpy as np
import pandas as pd

from nihai.notifier import chart_render
from nihai.notifier.telegram_bot import CommandListener, parse_command

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _FakeProvider:
    def __init__(self, n=80, empty=False):
        self.n, self.empty = n, empty

    def get_bars(self, symbol, interval, lookback):
        if self.empty:
            return pd.DataFrame()
        rng = np.random.default_rng(1)
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, size=self.n))
        idx = pd.date_range("2025-01-01", periods=self.n, freq="h")
        return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                             "close": close, "volume": rng.integers(1e3, 1e5, self.n)}, index=idx)


def test_render_price_png_returns_png(monkeypatch):
    monkeypatch.setattr(chart_render, "get_provider", lambda *a, **k: _FakeProvider())
    out = chart_render.render_price_png("aapl")
    assert isinstance(out, (bytes, bytearray)) and out[:8] == PNG_MAGIC
    assert len(out) > 1000          # gerçek bir görsel


def test_render_price_png_empty_symbol():
    assert chart_render.render_price_png("") is None


def test_render_price_png_no_data(monkeypatch):
    monkeypatch.setattr(chart_render, "get_provider", lambda *a, **k: _FakeProvider(empty=True))
    assert chart_render.render_price_png("AAPL") is None


def test_render_price_png_closes_figure_on_error(monkeypatch):
    """A4: figür oluştuktan SONRA hata (savefig patlar) olsa bile matplotlib figürü
    sızmamalı — render_price_png finally ile plt.close(fig) çağırır. get_fignums()
    çağrı öncesi/sonrası özdeş kalmalı (fix yoksa 1 fazla figür sızar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    monkeypatch.setattr(chart_render, "get_provider", lambda *a, **k: _FakeProvider())

    def _boom(*a, **k):
        raise RuntimeError("savefig patladı")
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _boom)

    before = set(plt.get_fignums())
    out = chart_render.render_price_png("AAPL")
    after = set(plt.get_fignums())
    assert out is None                    # hata yolu → sessiz None
    assert after == before                # sızan figür yok (finally kapattı)


def test_parse_command_grafik():
    assert parse_command("/grafik aapl") == ("grafik", "AAPL")
    assert parse_command("/chart@MyBot tsla") == ("chart", "TSLA")


def test_dispatch_routes_grafik_with_uppercased_arg():
    seen = {}
    def fake(arg):
        seen["arg"] = arg
        return None          # foto gönderildi varsayımı → metin yok
    cl = CommandListener({"grafik": fake}, chat_id="1")
    assert cl.dispatch("/grafik msft") is None
    assert seen["arg"] == "MSFT"


def test_dispatch_missing_arg_returns_usage():
    def grafik(arg):
        return "Kullanım: /grafik SEMBOL" if not arg else None
    cl = CommandListener({"grafik": grafik}, chat_id="1")
    assert "Kullanım" in cl.dispatch("/grafik")
