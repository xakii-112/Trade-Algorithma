"""
chart_render.py — Telegram /grafik için sunucu-tarafı PNG fiyat grafiği (B2).

Headless matplotlib (Agg) ile sembolün son barlarını kapanış çizgisi + EMA20/EMA50 +
hacim alt-paneli olarak çizer, PNG bayt dizisi döndürür. telegram_bot.send_photo ile
gönderilir. Web panodaki canlı lightweight-charts grafiğinin (tarayıcı) Telegram karşılığı.

SİNYAL-ONLY / DÜRÜSTLÜK: yalnız geçmiş fiyatı ve iki ortalamayı gösterir — kestirim, hedef
veya 'olması gereken fiyat' çizmez (o, web panonun /api/analyze işi). Veri yoksa None döner.
"""
from __future__ import annotations

import io

from loguru import logger

from nihai.config import DATA_PROVIDER
from nihai.data.data_provider import Interval, get_provider


def render_price_png(symbol: str, *, interval: Interval = "1h", lookback: int = 120,
                     df=None) -> bytes | None:
    """Sembol için kapanış + EMA20/50 + hacim PNG'si. Hata/veri yoksa None (sessiz).

    df verilirse (örn. chart_analyst aynı barları hem çizer hem yorumlatır) yeniden
    veri çekilmez; verilmezse sağlayıcıdan alınır (mevcut davranış)."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return None
    fig = None          # A4: hata yolunda da kapatabilmek için (aşağıda finally)
    try:
        if df is None:
            provider = get_provider(DATA_PROVIDER, with_cache=True)
            df = provider.get_bars(sym, interval, max(40, min(int(lookback), 400)))
        if df is None or getattr(df, "empty", True) or len(df) < 20:
            logger.debug(f"[chart] {sym} için yeterli bar yok")
            return None

        import matplotlib
        matplotlib.use("Agg")          # GUI yok → daemon thread'de güvenli
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        from nihai.signals.strategy import ema

        close = df["close"].astype(float)
        e20 = ema(close, 20)
        e50 = ema(close, 50)
        x = df.index
        last = float(close.iloc[-1])
        first = float(close.iloc[0])
        chg = (last - first) / first * 100.0 if first else 0.0
        up = chg >= 0
        line_c = "#16c784" if up else "#ea3943"

        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(8, 4.8), dpi=110, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#0d1124")

        for a in (ax, axv):
            a.set_facecolor("#0d1124")
            a.grid(True, color="#ffffff10", linewidth=0.6)
            a.tick_params(colors="#94a3b8", labelsize=8)
            for s in a.spines.values():
                s.set_color("#ffffff20")

        ax.plot(x, close, color=line_c, linewidth=1.6, label="Kapanış")
        ax.plot(x, e20, color="#f0b90b", linewidth=1.0, label="EMA20")
        ax.plot(x, e50, color="#22d3ee", linewidth=1.0, label="EMA50")
        ax.axhline(last, color=line_c, linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(f"{sym}  ·  {last:.2f}  ({chg:+.2f}%)  ·  {interval}",
                     color="#f8fafc", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7, facecolor="#161b2e",
                  edgecolor="#ffffff20", labelcolor="#cbd5e1")

        if "volume" in df.columns:
            vol = df["volume"].astype(float).fillna(0)
            axv.bar(x, vol, color="#3b4252", width=0.8 * (x[1] - x[0]) if len(x) > 1 else 0.02)
            axv.tick_params(labelleft=False)
        axv.set_ylabel("Hacim", color="#64748b", fontsize=7)

        try:
            axv.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        except Exception:
            pass
        fig.autofmt_xdate(rotation=0, ha="center")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"[chart] {sym} grafik üretilemedi: {e}")
        return None
    finally:
        # A4: başarı VEYA hata farketmeksizin oluşturulmuş figürü kapat (bellek/handle
        # sızıntısı önlenir). fig None ise (hata figür kurulmadan oluştu) atlanır; fig
        # atandıysa plt import'u zaten başarılıydı → yerel 'plt' ismi bağlıdır.
        if fig is not None:
            plt.close(fig)
