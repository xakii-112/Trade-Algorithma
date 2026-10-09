"""
xsect.py — Cross-sectional (kıyaslamalı) momentum portföy stratejisi + ölçüm.

Fikir (Jegadeesh-Titman momentum anomalisi): hisseleri tek tek zamanlamak yerine,
her rebalance'ta evreni geçmiş getiriye göre SIRALA, en güçlü K'yı eşit-ağırlık tut,
periyodik yenile. "Güçlü kalan güçlü kalır" — fiyat-göstergesi zamanlamasından
(EDGE_FINDINGS'te mega-cap'te kenarsız çıktı) FARKLI bir kaynak.

Klasik "lookback-skip" momentum: skor = getiri(t-lookback → t-skip). Son `skip` bar
atlanır (kısa-vade dönüş/reversal gürültüsünü ekarte eder; "12-1" momentum'un mantığı).

Look-ahead güvenli: sıralama yalnızca rebalance gününe (dahil) kadarki fiyatla; tutma
getirisi rebalance kapanışından bir sonraki rebalance kapanışına ölçülür.

KRİTİK kontrol (EDGE_FINDINGS §6): momentum, evrenin EŞİT-AĞIRLIK AL-TUT'unu aşmalı.
Aşmıyorsa sadece piyasa betasını biniyordur, kenar değil. İki periyotta da aşmalı.

    python xsect.py                                   # nasdaq100, 1d, 2y+5y, top 10
    python xsect.py --periods 2y,5y --top 10 --lookback 126 --skip 21 --hold 21
    python xsect.py --top 15 --lookback 252 --skip 21   # 12-1 momentum, top 15
"""
from __future__ import annotations

import argparse
import statistics
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import numpy as np
import pandas as pd
from loguru import logger

from nihai.config import CONFIG
from nihai.execution.backtest import load_bars
from nihai.universe import BASKET

# Tur-52: PIT (survivorship-temiz) evren — yerel disk, ağ YOK.
PIT_GLOB = "data/alpaca_hist/*_1d_all.csv"


def pit_symbols_as_of(as_of: pd.Timestamp) -> list[str]:
    """`as_of` tarihinde S&P 500'de OLAN sembollerin yerel fiyat dosyası olanları.

    Neden Wikipedia'nın değişiklik tablosu değil: o tablo `_SP500_URL` sayfasından
    KALKMIŞ (2026-08 itibarıyla `tabs[1]` bir navbox) → `pit_universe.sp500_as_of`
    KeyError veriyor. Ama gereken bilgi zaten DİSKTE: `data/alpaca_hist/*_1d_all.csv`
    kümesi Tur-46 rebalans pilotunun `sp500_as_of('2021-01-01')` evreninden üretildi ve
    **bugün endekste olmayan ~90 ismi içerir** (45'inin fiyat serisi evren sonundan >90
    gün önce biter — doğrudan ölçüldü).

    Kalan tek kirlilik İLERİ-BAKIŞTIR: `as_of`'tan SONRA endekse giren isimler de dosya
    kümesinde durur ve 2020'den beri fiyatları vardır — yani "sonradan büyüyüp endekse
    girdiği için seçilmiş" isimler. Onlar mevcut üyelik tablosunun `Date added`
    kolonuyla ELENİR (o kolon hâlâ ayakta). Kalıntı yanlılık: `as_of` ile yerel anlık
    görüntü (2021-01-01) arasında ÇIKARILAN isimler bir süre fazladan taşınır — küçük
    ve momentum LEHİNE DEĞİL (ölen isimler evrende kalır).
    """
    import glob
    import os
    yerel = {os.path.basename(f).split("_1d_all")[0] for f in glob.glob(PIT_GLOB)}
    try:
        from nihai.data.pit_universe import _SP500_URL, _norm, _read_html_tables
        cur = _read_html_tables(_SP500_URL)[0]
        eklendi = pd.to_datetime(cur["Date added"], errors="coerce")
        gec = {_norm(s) for s, d in zip(cur["Symbol"], eklendi, strict=False)
               if pd.notna(d) and d > as_of}
    except Exception as e:                      # ağ/şema kırılırsa ölçüm durmaz, DARALMAZ
        logger.warning(f"[xsect] mevcut üyelik tablosu okunamadı ({e}) — "
                       f"ileri-bakış elemesi YAPILAMADI, sonuç momentum LEHİNE yanlıdır")
        gec = set()
    syms = sorted(yerel - gec)
    logger.info(f"[xsect] PIT @ {as_of.date()}: {len(yerel)} yerel dosya, "
                f"{len(gec & yerel)}'i sonradan eklendiği için ELENDİ → {len(syms)} sembol")
    return syms


def build_panel_pit(symbols: list[str], start: pd.Timestamp) -> pd.DataFrame:
    """Yerel `_1d_all` (düzeltilmiş) kapanışlarını panele topla — ağ/throttle YOK."""
    series = {}
    for sym in symbols:
        path = f"data/alpaca_hist/{sym}_1d_all.csv"
        try:
            df = pd.read_csv(path, usecols=["timestamp", "close"])
        except (OSError, ValueError):
            continue
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["ts", "close"])
        df = df[df["ts"] >= start]
        if len(df) > 60:
            series[sym] = df.set_index("ts")["close"]
    return pd.DataFrame(series).sort_index()


def build_panel(symbols: list[str], period: str, interval: str = "1d",
                throttle: float = 0.2) -> pd.DataFrame:
    """Sembollerin kapanışlarını ortak tarih indeksinde hizalanmış panele topla."""
    import time
    series = {}
    n = len(symbols)
    for i, s in enumerate(symbols, 1):
        try:
            df = load_bars(s, interval, period=period)
            if not df.empty and len(df) > 60:
                series[s] = df["close"]
        except Exception as e:
            logger.debug(f"[xsect] {s}: {e}")
        if i % 20 == 0:
            logger.info(f"[xsect] {i}/{n} sembol yüklendi")
        if throttle > 0:
            time.sleep(throttle)
    panel = pd.DataFrame(series).sort_index()
    return panel


def run_momentum(panel: pd.DataFrame, *, lookback: int, skip: int, hold: int,
                 top: int, cost_pct: float) -> dict:
    """Cross-sectional momentum backtest. Eşit-ağırlık top-K, her `hold` barda rebalance.

    Dönüş: equity eğrisi, tutma-dönemi getirileri (her sembol-dönem bir 'trade'),
    portföy metrikleri + eşit-ağırlık al-tut (beta kontrolü)."""
    dates = panel.index
    n = len(dates)
    first = lookback + 1  # ilk rebalance: yeterli geçmiş için
    if n < first + hold:
        return {}

    equity = [1.0]
    period_returns: list[float] = []   # her rebalance'ta portföy getirisi
    bh_period_returns: list[float] = []  # aynı ızgarada eşit-ağırlık EVREN getirisi
    trade_returns: list[float] = []    # her sembol-dönem (per-position) getiri
    prev_holdings: set[str] = set()
    n_rebal = 0

    t = first
    while t + hold < n:
        # Skor: getiri (t-lookback → t-skip), look-ahead-güvenli (hepsi ≤ t)
        p_now = panel.iloc[t - skip]
        p_then = panel.iloc[t - lookback]
        score = (p_now / p_then - 1.0)
        # Geçerli (ikisi de dolu) semboller
        valid = score.dropna()
        # t anında ve t+hold anında fiyatı olan semboller (tutma getirisi ölçülebilir)
        entry_px = panel.iloc[t]
        exit_px = panel.iloc[t + hold]
        tradable = [s for s in valid.index
                    if pd.notna(entry_px.get(s)) and pd.notna(exit_px.get(s))
                    and entry_px.get(s, 0) > 0]
        if not tradable:
            t += hold
            continue
        ranked = valid[tradable].sort_values(ascending=False)
        holdings = list(ranked.index[:top])
        if not holdings:
            t += hold
            continue

        # Tutma getirisi: eşit-ağırlık, rebalance kapanışından sonrakine
        rets = [(exit_px[s] / entry_px[s] - 1.0) for s in holdings]
        trade_returns.extend(rets)
        port_ret = float(np.mean(rets))
        # AYNI ızgarada eşit-ağırlık EVREN getirisi (risk-düzeltilmiş kıyas için).
        # `edge_vs_buyhold` yalnız TOPLAM getiriyi kıyaslar; 10 isimlik yoğun bir
        # portföy 525 isimlik bir sepetten zaten daha oynak olacağı için tek başına
        # aldatıcıdır — beta'yı aşmak ile RİSK BİRİMİ BAŞINA aşmak ayrı sorulardır.
        bh_period_returns.append(float(np.mean(
            [(exit_px[s] / entry_px[s] - 1.0) for s in tradable])))

        # İşlem maliyeti: devir oranı (değişen isimler) × slipaj × 2 (round-trip)
        new_set = set(holdings)
        turnover = len(new_set ^ prev_holdings) / max(2 * top, 1)
        port_ret -= turnover * cost_pct * 2
        prev_holdings = new_set

        period_returns.append(port_ret)
        equity.append(equity[-1] * (1 + port_ret))
        n_rebal += 1
        t += hold

    # Eşit-ağırlık al-tut beta kontrolü: aynı pencerede tüm evren ortalama getiri.
    #
    # Tur-52 DÜZELTMESİ — uç NOKTA değil, uç GÖZLEM. Eski hâli `panel.iloc[first]` ve
    # `panel.iloc[t]` satırlarını okuyordu; bu yalnız TÜM sembollerin aynı gün başlayıp
    # aynı gün bittiği (yfinance) panelde doğrudur. PIT panelinde isimler ÖLÜR ve yerel
    # dosyalar farklı günlerde biter: son satırda 525 sembolün yalnız 4'ünde fiyat vardı
    # → `bh_rets` BOŞ → kontrol sessizce **0.0** döndü ve momentum "beta'yı aştı" gibi
    # göründü. Kontrolün kendisi kaybolduğunda §6'nın tuzağı geri gelir.
    #
    # Doğrusu: her sembol için pencere İÇİNDEKİ ilk ve son geçerli fiyat. Ölen isim
    # son işlem fiyatıyla katılır (kaybı SAYILIR) — momentum bacağı da aynı paneli
    # kullandığı için muamele simetriktir. Dolu panelde ilk/son geçerli = ilk/son satır,
    # yani eski davranışla bit-denk (test bunu çapalar).
    win = panel.iloc[first:t + 1]
    bh_rets = []
    for s in panel.columns:
        col = win[s].dropna()
        if len(col) >= 2 and col.iloc[0] > 0:
            bh_rets.append(float(col.iloc[-1]) / float(col.iloc[0]) - 1.0)
    buyhold = float(np.mean(bh_rets)) * 100 if bh_rets else 0.0

    # Metrikler
    total_ret = (equity[-1] - 1.0) * 100
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    gw = sum(wins)
    gl = abs(sum(losses))
    # Yıllık Sharpe (rebalance frekansından): hold bar = 1 dönem
    ppy = 252 / hold
    sharpe = 0.0
    if len(period_returns) > 1:
        sd = statistics.pstdev(period_returns)
        if sd > 0:
            sharpe = (statistics.mean(period_returns) / sd) * (ppy ** 0.5)
    # Max drawdown
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)

    def _sharpe(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        sd = statistics.pstdev(xs)
        return round((statistics.mean(xs) / sd) * (ppy ** 0.5), 2) if sd > 0 else 0.0

    return {
        "rebalances": n_rebal,
        "positions": len(trade_returns),
        "period_win_rate_pct": round(100 * sum(1 for r in period_returns if r > 0)
                                     / len(period_returns), 1) if period_returns else 0.0,
        "position_win_rate_pct": round(100 * len(wins) / len(trade_returns), 1) if trade_returns else 0.0,
        "profit_factor": round(gw / gl, 2) if gl > 0 else None,
        "avg_position_ret_pct": round(100 * statistics.mean(trade_returns), 2) if trade_returns else 0.0,
        "total_return_pct": round(total_ret, 1),
        "buyhold_return_pct": round(buyhold, 1),
        "edge_vs_buyhold_pct": round(total_ret - buyhold, 1),
        "sharpe_annualized": round(sharpe, 2),
        "buyhold_sharpe": _sharpe(bh_period_returns),
        "max_drawdown_pct": round(100 * mdd, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-sectional momentum (vs al-tut betası)")
    ap.add_argument("--universe", default="nasdaq100",
                    choices=["basket", "nasdaq100", "sp500", "pit_local"],
                    help="pit_local = survivorship-TEMİZ (ölmüş isimler dahil, yerel disk)")
    ap.add_argument("--periods", default="2y,5y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--lookback", type=int, default=126, help="momentum penceresi (bar)")
    ap.add_argument("--skip", type=int, default=21, help="atlanan son bar (kısa-vade reversal)")
    ap.add_argument("--hold", type=int, default=21, help="rebalance aralığı (bar)")
    ap.add_argument("--top", type=int, default=10, help="tutulan en güçlü K sembol")
    args = ap.parse_args()

    pit = args.universe == "pit_local"
    if args.universe == "basket":
        symbols = BASKET
    elif pit:
        symbols = []          # periyot başına yeniden kurulur (üyelik as-of'a bağlı)
    else:
        from nihai.execution.screen import nasdaq100_tickers, sp500_tickers
        symbols = nasdaq100_tickers() if args.universe == "nasdaq100" else sp500_tickers()

    cost = CONFIG.backtest.stock_slippage_pct
    periods = [p.strip() for p in args.periods.split(",")]
    print(f"=== Cross-sectional momentum | {args.universe} "
          f"({'periyot başına PIT' if pit else len(symbols)}) | "
          f"{args.interval} | lookback={args.lookback} skip={args.skip} "
          f"hold={args.hold} top={args.top} ===")
    print("Hedef: edge_vs_buyhold > 0 İKİ periyotta da (momentum, beta'yı aşmalı).\n")

    for per in periods:
        if pit:
            yil = float(str(per).lower().rstrip("y") or 0)
            start = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.DateOffset(years=int(yil))
            panel = build_panel_pit(pit_symbols_as_of(start), start)
        else:
            panel = build_panel(symbols, per, args.interval)
        if panel.empty:
            print(f"--- {per}: veri yok ---\n")
            continue
        m = run_momentum(panel, lookback=args.lookback, skip=args.skip, hold=args.hold,
                         top=args.top, cost_pct=cost)
        print(f"{'='*56}\n--- PERİYOT: {per} ({panel.shape[1]} sembol, {panel.shape[0]} bar) ---\n{'='*56}")
        if not m:
            print("  (yeterli veri yok)\n")
            continue
        for k, v in m.items():
            print(f"  {k:<24} {v}")
        verdict = ("KENAR VAR ✓" if m["edge_vs_buyhold_pct"] > 0 else "sadece beta (kenar yok)")
        print(f"  → {verdict}\n")


if __name__ == "__main__":
    main()
