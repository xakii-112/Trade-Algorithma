"""
terminal_dashboard.py — rich ile canlı terminal panosu.

PDF Referansları:
- Madde 39 : Boşta & Pozisyonda pano standartları
- Madde 22 : Seans durumu (RTH/kapalı/kapanışa kalan)
- Madde 10 : Günlük risk durumu (PDT, max loss)
"""
from __future__ import annotations

import re
import threading
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import nihai.execution.watchlist as watchlist_store
from nihai.config import CONFIG, RUN_MODE
from nihai.data.market_calendar import is_market_open, session_remaining
from nihai.risk.risk_manager import PortfolioState

console = Console()

# AI dürüstlük şeridi — 60 sn TTL'li tek satır önbelleği.
# NEDEN ÖNBELLEK: `LiveDashboard` saniyede 2 kez render eder; kalibrasyon artefaktı ise
# günde bir kez tazelenir. Her karede JSON okumak saniyede 2 disk erişimi demekti — üstelik
# OneDrive üstünde. Şerit YAVAŞ değişen bir büyüklüktür, hızı ondan alır.
_AI_STRIP: tuple[float, str] = (0.0, "")
_AI_STRIP_TTL = 60.0


def _ai_strip() -> str:
    """Kalibrasyonun tek satırlık dürüstlük şeridi (KARAR-DIŞI, SALT OKUNUR).

    `nihai/ai` katmanı panoya ilk kez burada bağlanır (Tur-37). Neden BURASI ve neden
    yalnız BU satır: pano operatörün karar YÜZEYİDİR; oraya ikinci bir olasılık/yön sütunu
    koymak onu sinyal sanma riski taşır (neural_shadow'un panoya çıkmama gerekçesiyle
    AYNI). Kalibrasyon şeridi bir tahmin değil, sistemin KENDİ tahminleri hakkındaki
    dürüstlük ölçüsüdür — operatörün görmesi gereken tam olarak budur.

    Hata/AI katmanı yok → boş string (pano hiçbir koşulda düşmez)."""
    global _AI_STRIP
    now = datetime.now().timestamp()
    if now - _AI_STRIP[0] < _AI_STRIP_TTL and _AI_STRIP[1]:
        return _AI_STRIP[1]
    try:
        from nihai.ai.observatory import calibration_line
        line = calibration_line()
    except Exception:
        line = ""
    _AI_STRIP = (now, line)
    return line


def _flat_panel(state: PortfolioState, iv_rank: float | None = None,
                event_calendar: list[str] | None = None) -> Panel:
    open_market = is_market_open()
    rem = session_remaining()
    rem_str = f"{rem.seconds//3600}s {(rem.seconds%3600)//60}dk" if rem else "kapalı"

    risk_used = sum(p.risk_dollars for p in state.open_positions)
    gross_pct = risk_used / state.equity * 100 if state.equity else 0

    pdt_active = (CONFIG.regulation.pdt_enabled
                  and state.equity < CONFIG.regulation.pdt_threshold_equity)
    if not CONFIG.regulation.pdt_enabled:
        pdt_line = "DEVRE DISI (pdt_enabled=False)"
    elif pdt_active:
        pdt_line = (f"AKTIF (NLV < ${CONFIG.regulation.pdt_threshold_equity:,.0f}) - "
                    f"Gun-ici islem: {state.day_trades_count} / "
                    f"{CONFIG.regulation.pdt_max_day_trades_under_threshold}")
    else:
        pdt_line = "PASIF (NLV yeterli)"

    _dml = CONFIG.risk.daily_max_loss_pct
    if _dml <= 0:
        daily_status, daily_limit_txt = "DEVRE DISI", "fren kapali"
    else:
        daily_status = ("GUVENLI" if state.daily_pnl > -state.equity * _dml else "STOP")
        daily_limit_txt = f"Aktif Limit: %{_dml*100:.1f}"

    body = Table.grid(padding=(0, 1))
    body.add_column(style="bold cyan", width=22)
    body.add_column(style="white")
    # Tur-106 (BULGU-106-25): NLV = gerçekleşmiş özkaynak + açık K/Z (pano /api/status ile aynı tanım).
    body.add_row("Toplam Deger (NLV)", f"{state.equity + state.open_pnl:,.2f} USD "
                                       f"(gerceklesmis {state.equity:,.2f})")
    body.add_row("Buying Power (Reg T)", f"{state.equity * 2:,.2f} USD")
    body.add_row("Gunluk Net PnL", f"{state.daily_pnl:+,.2f} USD "
                                    f"({state.daily_pnl/state.equity*100:+.2f}%)")
    body.add_row("Gunluk Risk Durumu", f"{daily_status} ({daily_limit_txt})")
    body.add_row("Portfoy Maruziyeti",
                 f"{len(state.open_positions)} / {CONFIG.portfolio.max_open_positions} pozisyon | "
                 f"Brut risk %{gross_pct:.1f} (tavan %{CONFIG.portfolio.max_gross_risk_pct*100:.0f})")
    body.add_row("PDT Durumu", pdt_line)
    body.add_row("Seans Durumu", f"{'ACIK (RTH)' if open_market else 'KAPALI'} - Kapanisa {rem_str}")
    body.add_row("Genel Sistem Modu", "BEKLEMEDE (Sinyal araniyor)")
    if iv_rank is not None:
        rejim = "DUSUK -> prim alimi favori" if iv_rank < CONFIG.strategy.iv_rank_low_threshold \
                else "YUKSEK -> prim satma favori" if iv_rank > CONFIG.strategy.iv_rank_high_threshold \
                else "ORTA"
        body.add_row("IV Rank (watchlist)", f"{iv_rank:.0f} ({rejim})")
    if event_calendar:
        body.add_row("Olay Takvimi (48s)", " | ".join(event_calendar))
    _wl = watchlist_store.current()
    _wl_str = ', '.join(_wl[:15]) + (f" (+{len(_wl)-15})" if len(_wl) > 15 else "")
    body.add_row("Calisma Kipi", f"{RUN_MODE} | Watchlist({len(_wl)}): {_wl_str}")
    _pins, _blocks = watchlist_store.pins(), watchlist_store.blocks()
    if _pins or _blocks:
        body.add_row("Watchlist Manuel",
                     f"pin: {', '.join(_pins) or '-'} | block: {', '.join(_blocks) or '-'}")
    _ls = watchlist_store.last_scan()
    if _ls:
        body.add_row("Son Tarama", _ls)
    _ai = _ai_strip()
    if _ai:
        body.add_row("AI Golge (karar-disi)", _ai)

    return Panel(body, title="[bold yellow]PORTFOY VE GUNLUK RISK DURUMU[/]",
                 subtitle=f"[dim]{datetime.now():%Y-%m-%d %H:%M:%S}[/]",
                 border_style="cyan")


def _position_panel(state: PortfolioState) -> Panel:
    if not state.open_positions:
        return Panel(Text("Açık pozisyon yok.", style="dim"), title="AKTIF POZISYON TAKIP",
                     border_style="dim")

    t = Table.grid(padding=(0, 1))
    t.add_column(style="bold magenta", width=22)
    t.add_column(style="white")

    for p in state.open_positions:
        if p.asset_type == "option":
            label = f"{p.symbol} {p.expiry} {(p.option_type or '?').upper()} {p.strike} ({p.side})"
            t.add_row("Kontrat", label)
            t.add_row("Giris / Adet", f"{p.entry_price:.2f} USD x {p.quantity} kontrat")
            if p.greeks:
                g = p.greeks
                t.add_row("Greekler", f"Δ{g.delta:+.2f} | Γ{g.gamma:+.3f} | Θ{g.theta:+.2f} | ν{g.vega:+.2f}")
                t.add_row("IV", f"%{g.iv*100:.1f}")
        else:
            t.add_row("Hisse", f"{p.symbol} ({p.side} x {p.quantity})")
            t.add_row("Giris / Stop", f"{p.entry_price:.2f} → stop {p.stop:.2f}")
        t.add_row("Risk", f"{p.risk_dollars:,.2f} USD")
        t.add_row("─" * 20, "─" * 30)

    return Panel(t, title="[bold green]AKTIF POZISYON TAKIP EKRANI[/]", border_style="green")


def _recent_signals_panel(recent: list[str]) -> Panel:
    if not recent:
        return Panel(Text("Henüz sinyal yok.", style="dim"), title="SON SİNYALLER", border_style="dim")
    body = "\n".join(recent[-10:])
    return Panel(body, title="[bold]SON SİNYALLER (En son 10)[/]", border_style="blue")


def build_dashboard(state: PortfolioState, recent_signals: list[str] | None = None,
                    iv_rank: float | None = None) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(_flat_panel(state, iv_rank=iv_rank), name="top", size=15),  # +1: AI şeridi
        Layout(_position_panel(state), name="mid", size=12),
        Layout(_recent_signals_panel(recent_signals or []), name="bottom"),
    )
    return layout


def render_once(state: PortfolioState, recent_signals: list[str] | None = None,
                iv_rank: float | None = None) -> None:
    console.print(build_dashboard(state, recent_signals, iv_rank))


# "YYYY-AA-GG SS:DD:SS | …" — Tur-106 (BULGU-106-33). Satırlar state.json'da kalıcıdır; yalnız saat taşıyan eski
# biçim günler sonra da güncel görünüyordu (3 Ekim'de panonun tepesinde 22 Eylül kapanışları duruyordu).
TARIHLI_SATIR = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ")


def tarihli_satirlar(satirlar: list[str]) -> list[str]:
    """Yalnız tarih damgalı satırlar (eski, yalnız-saat biçimi atılır: hangi güne ait olduğu bilinemez)."""
    return [s for s in satirlar if isinstance(s, str) and TARIHLI_SATIR.match(s)]


class LiveDashboard:
    """Canlı yenilemeli pano (main loop'ta kullan)."""

    def __init__(self):
        self.recent_signals: list[str] = []
        self._live: Live | None = None
        self._kilit = threading.Lock()   # tarama iş parçacıkları da satır ekler; rich Live'a DOKUNMAZ

    def __enter__(self):
        # İlk render boş — refresh ile dolacak.
        self._live = Live(Text(""), console=console, refresh_per_second=2, screen=False)
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)

    def refresh(self, state: PortfolioState, iv_rank: float | None = None) -> None:
        if self._live:
            self._live.update(build_dashboard(state, self.recent_signals, iv_rank))

    def add_signal_line(self, line: str) -> None:
        with self._kilit:
            self.recent_signals = [*self.recent_signals, f"{datetime.now():%Y-%m-%d %H:%M:%S} | {line}"][-20:]
