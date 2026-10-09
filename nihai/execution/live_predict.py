"""
live_predict.py — Seçili (≤30) hisse için SÜREKLI sinyal/öngörü monitörü.

Her döngüde her sembol için: canlı bar (DATA_PROVIDER, örn. alpaca_stream) → generate_signal
→ aksiyon (LONG/SHORT/WAIT) + giriş/stop/hedef/RR/kalite + meta-model KAZANMA OLASILIĞI
(öngörü/güven). Tablo sürekli yenilenir.

DÜRÜST ÇERÇEVE:
- Bu SİNYAL-ONLY: emir GÖNDERMEZ. Gerçek emir için broker/order katmanı (kasıtlı yok) +
  KANITLANMIŞ kenar gerekir. EDGE_FINDINGS: mevcut sinyaller survivorship-temiz testte
  marjinal → P(win)'i "kesin tahmin" değil, model güveni olarak oku.
- P(win) = meta_gate modeli çıktısı (data/meta_model.json). Model yoksa "—".

    python -m nihai.execution.live_predict                         # STREAM_SYMBOLS, HTF_INTERVAL, 60s
    python -m nihai.execution.live_predict --symbols AAPL,MSFT,NVDA --interval 1h --every 30
    python -m nihai.execution.live_predict --once                  # tek tur, çık
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from datetime import datetime

if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from nihai.config import HTF_INTERVAL
from nihai.data.data_provider import get_provider
from nihai.data.market_calendar import kapanmis_gunluk
from nihai.runtime_strategy import apply as _strateji_sec
from nihai.signals.meta_gate import gate_health, get_gate
from nihai.signals.quant import CANLI_GUNLUK_BAR
from nihai.signals.strategy import SignalAction, generate_signal, is_quant_label


def _symbols(arg: str | None) -> list[str]:
    if arg:
        return [s.strip().upper() for s in arg.split(",") if s.strip()]
    env = [s.strip().upper() for s in os.getenv("STREAM_SYMBOLS", "").split(",") if s.strip()]
    if env:
        return env
    from nihai.universe import BASKET  # makul varsayılan (16 likit isim) — kendi 30'unu STREAM_SYMBOLS'la ver
    return BASKET


def _row(symbol: str, provider, gate, interval: str) -> dict:
    df = provider.get_bars(symbol, interval, lookback_bars=220)
    if df is None or df.empty or len(df) < 60:
        return {"symbol": symbol, "action": "NODATA", "price": 0, "rr": 0,
                "q": 0, "pwin": None, "why": "veri yok"}
    # Tur-106 (BULGU-106-27): canlı taramayla AYNI strateji kaynağı ve girdiler. quant_trend günlük (kapanmış)
    # barla karar verir; onlar geçilmeden gün-içi aralıkta her satır QUANT_GUNLUK_BAR_GEREKLI olurdu.
    strateji = _strateji_sec()
    df_gunluk = (kapanmis_gunluk(provider.get_bars(symbol, "1d", lookback_bars=CANLI_GUNLUK_BAR))
                 if strateji == "quant_trend" else None)
    sig = generate_signal(symbol, df, strategy_mode=strateji, df_daily=df_gunluk)
    pwin = None
    # P(win) yalnızca GERÇEK giriş sinyalinde anlamlı: WAIT'te giriş/rr/kalite yok →
    # model dejenere özelliklerle saçmalar (≈%100). Yanıltmamak için sadece LONG/SHORT.
    # Meta kapı quant sinyalinde kapsam dışıdır (BULGU-97-4) → P(win) "—".
    if (sig.action in (SignalAction.LONG, SignalAction.SHORT)
            and not is_quant_label((sig.metadata or {}).get("strategy"))):
        with contextlib.suppress(Exception):
            pwin = gate.probability(df, sig)   # META_GATE bayrağından bağımsız: göster
    why = sig.reasons[0] if sig.reasons else ""
    return {"symbol": symbol, "action": sig.action.value, "price": float(sig.price or 0),
            "stop": float(sig.stop or 0), "target": float(sig.target or 0),
            "rr": float(sig.rr_ratio or 0), "q": float(sig.quality_score or 0),
            "pwin": pwin, "why": why}


_ORDER = {"LONG": 0, "SHORT": 1, "WAIT": 2, "NODATA": 3}


def _print_table(rows: list[dict], interval: str, src: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    rows = sorted(rows, key=lambda r: (_ORDER.get(r["action"], 9),
                                       -(r["pwin"] if r["pwin"] is not None else -1)))
    print(f"\n{'='*78}")
    print(f" CANLI ÖNGÖRÜ  {ts}  | interval={interval} | kaynak={src} | SİNYAL-ONLY (emir yok)")
    print(f"{'='*78}")
    print(f" {'SEMBOL':<7}{'AKSİYON':<8}{'FİYAT':>9}{'RR':>6}{'KALİTE':>8}{'P(win)':>8}  GEREKÇE")
    print(f" {'-'*76}")
    for r in rows:
        pw = f"{r['pwin']*100:4.0f}%" if r.get("pwin") is not None else "  — "
        act = r["action"]
        mark = "▲" if act == "LONG" else ("▼" if act == "SHORT" else " ")
        print(f" {r['symbol']:<7}{mark}{act:<7}{r['price']:>9.2f}{r['rr']:>6.2f}"
              f"{r['q']:>8.2f}{pw:>8}  {r['why'][:34]}")
    actionable = [r for r in rows if r["action"] in ("LONG", "SHORT")]
    print(f" {'-'*76}")
    print(f" {len(actionable)} aksiyon sinyali / {len(rows)} sembol"
          + (f"  → en güçlü: {actionable[0]['symbol']} "
             f"P(win)={actionable[0]['pwin']*100:.0f}%" if actionable and actionable[0]['pwin'] is not None else ""))


def main():
    ap = argparse.ArgumentParser(description="Sürekli sinyal/öngörü monitörü (≤30 hisse)")
    ap.add_argument("--symbols", default=None, help="virgülle; boşsa STREAM_SYMBOLS/BASKET")
    ap.add_argument("--interval", default=HTF_INTERVAL)
    ap.add_argument("--every", type=float, default=60.0, help="tur arası saniye")
    ap.add_argument("--once", action="store_true", help="tek tur, çık")
    args = ap.parse_args()

    symbols = _symbols(args.symbols)
    provider = get_provider()   # DATA_PROVIDER env (alpaca_stream önerilir)
    gate = get_gate()
    src = getattr(provider, "name", "?")
    print(f"İzlenen {len(symbols)} sembol: {', '.join(symbols)}")
    print(f"Veri kaynağı: {src} | meta-model: "
          f"{'AKTİF ('+gate.kind+')' if gate.active else 'YOK (P(win)=—)'}")
    if len(symbols) > 30:
        print("UYARI: 30'dan fazla sembol — ücretsiz IEX WS limiti ~30; fazlası REST'e düşer.")

    try:
        while True:
            rows = [_row(s, provider, gate, args.interval) for s in symbols]
            _print_table(rows, args.interval, src)
            # Tur-44: açılış satırı "AKTİF (logistic)" der ama bu yalnız MODEL YÜKLENDİ
            # demektir. Kapı her soruda hata veriyorsa P(win) sütunu boş kalır ve
            # operatör bunu "model yok" sanır — durumu SÖYLE.
            _h = gate_health()
            if _h["durum"] in ("SUSUYOR", "KISMI_HATA"):
                print(f"UYARI: meta-kapı {_h['durum']} — {_h['failures']}/{_h['calls']} "
                      f"soruda hata. Son: {_h['last_error']}")
            if args.once:
                break
            time.sleep(max(5.0, args.every))
    except KeyboardInterrupt:
        print("\nDurduruldu.")
    finally:
        stop = getattr(provider, "stop", None)
        if callable(stop):
            with contextlib.suppress(Exception):
                stop()


if __name__ == "__main__":
    main()
