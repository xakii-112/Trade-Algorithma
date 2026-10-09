"""
option_strategy.py — Dayanak yön sinyalini opsiyon kontratına çevirir.

PDF Referansları:
- Madde 2  : Long opsiyon → primin tamamı risk
- Madde 16 : IV rejimi (IV çok yüksekse skip — ileride credit spread)
- Madde 17 : IV crush koruması (planlı olay öncesi şişmiş IV'de long alma)
- Madde 21 : Likidite filtresi (OI/hacim/makas)
- Madde 27 : Greek limitleri (risk_manager.greek_caps_breach üzerinden)
- Madde 28 : DTE penceresi + pin riski
- Madde 29 : Uzlaşma stili metadata

Strateji: hedef delta'ya en yakın, en likit, makul DTE'de strike'ı seç.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from nihai.config import CONFIG
from nihai.data.data_provider import DataProvider
from nihai.data.event_calendar import ex_dividend_context, iv_crush_guard_for_option
from nihai.data.iv_history import append_iv_today, get_iv_rank
from nihai.risk.options_analyzer import (
    GreekSet,
    OptionType,
    compute_greeks,
    days_to_expiry,
    early_exercise_risk,
    filter_liquid_chain,
    settlement_style,
)
from nihai.signals.strategy import SignalAction


@dataclass
class OptionPick:
    """Seçilen opsiyon kontratı — sanal pozisyon açmak için tüm metadata."""
    symbol: str                # dayanak
    option_type: str           # 'c' | 'p'
    strike: float
    expiry: str                # YYYY-MM-DD
    bid: float
    ask: float
    mid: float
    last_price: float
    open_interest: int
    volume: int
    spread_pct: float
    greeks: GreekSet
    iv: float
    dte: int
    underlying_price: float
    settlement_info: dict = field(default_factory=dict)
    # Madde 25 (Tur-58) — ex-temettü erken-kullanım eşiği karnesi; boş sözlük = ölçülmedi.
    early_exercise: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        side = "Call" if self.option_type == "c" else "Put"
        return f"{self.symbol} {self.expiry} {side} {self.strike:.1f}"


def _score_contract(*, target_delta: float, contract_delta: float,
                    open_interest: float, spread_pct: float) -> float:
    """
    Yüksek skor = daha iyi kontrat.
      - Hedef delta'ya yakınlık (ana skor)
      - Likidite (OI log-scale)
      - Dar makas (penaltı)
    """
    delta_dist = abs(abs(contract_delta) - target_delta)
    # delta_dist=0 → +5.0; delta_dist=0.1 → +3.0; delta_dist=0.2 → +1.0
    delta_score = max(5.0 - delta_dist * 20.0, 0.0)
    liquidity_score = float(np.log1p(open_interest))   # OI=500 → ~6.2, OI=5000 → ~8.5
    spread_penalty = spread_pct * 50.0                  # %5 makas → -2.5 puan
    return delta_score + liquidity_score - spread_penalty


def select_option_contract(
    symbol: str,
    direction: SignalAction,
    provider: DataProvider,
    *,
    target_delta: float | None = None,
    dte_min: int | None = None,
    dte_max: int | None = None,
    risk_free_rate: float = 0.05,
) -> tuple[OptionPick | None, list[str]]:
    """
    Hisse sinyali (LONG/SHORT) → en uygun call/put kontratı.

    LONG  → call (delta hedef pozitif)
    SHORT → put  (delta hedef negatif, |delta| ile kıyas)

    Dönüş: (OptionPick or None, reasons[])
    """
    cfg = CONFIG.option_strategy
    target_delta = target_delta if target_delta is not None else cfg.target_delta
    dte_min = dte_min if dte_min is not None else cfg.dte_min
    dte_max = dte_max if dte_max is not None else cfg.dte_max

    if direction not in (SignalAction.LONG, SignalAction.SHORT):
        return None, [f"DIRECTION_{direction.value}_NOT_LONG_OR_SHORT"]

    # Madde 17 — IV crush guard (bilanço yakını → long opsiyon bloke)
    iv_crush_blocked, iv_crush_reason = iv_crush_guard_for_option(symbol)
    if iv_crush_blocked:
        return None, [iv_crush_reason]

    opt_type: OptionType = "c" if direction == SignalAction.LONG else "p"

    # 1) Tüm expiry listesini al + dayanak fiyatı
    first_chain = provider.get_option_chain(symbol)
    if not first_chain:
        return None, ["NO_CHAIN"]
    all_expiries = first_chain.get("all_expiries", [])
    underlying = float(first_chain.get("underlying_price") or 0)
    if underlying <= 0 or not all_expiries:
        return None, ["NO_UNDERLYING_OR_EXPIRIES"]

    # 2) Madde 28 — DTE penceresi filtresi
    candidates = [(exp, days_to_expiry(exp)) for exp in all_expiries]
    candidates = [(e, d) for e, d in candidates if dte_min <= d <= dte_max]
    if not candidates:
        return None, [f"NO_EXPIRY_IN_DTE_{dte_min}-{dte_max}"]

    settlement = settlement_style(symbol)

    best: OptionPick | None = None
    best_score = -1e9
    diag_reasons: list[str] = []

    for expiry, dte in candidates:
        chain_data = (first_chain if expiry == first_chain.get("expiry")
                      else provider.get_option_chain(symbol, expiry))
        if not chain_data:
            continue
        df = chain_data.get("calls" if opt_type == "c" else "puts")
        if df is None or df.empty:
            continue

        # 3) Madde 21 — likidite filtresi
        flt = filter_liquid_chain(
            df,
            min_oi=CONFIG.options.min_open_interest,
            min_volume=CONFIG.options.min_daily_volume,
            max_spread_pct=CONFIG.options.max_spread_pct_of_mid,
        )
        if flt.accepted.empty:
            diag_reasons.append(f"EXP_{expiry}_ALL_ILLIQUID")
            continue

        # 4) Her aday strike için Greekler + IV
        # Optimizasyon: BS implied-vol çözümü pahalı. Hedef delta dar bir moneyness
        # bandına denk geldiğinden, derin ITM/OTM strike'ları çözümden önce ele.
        acc = flt.accepted
        moneyness = (acc["strike"].astype(float) / underlying - 1.0).abs()
        acc = acc[moneyness <= 0.30]
        T = dte / 365.0
        for _, row in acc.iterrows():
            strike = float(row["strike"])
            bid = float(row["bid"])
            ask = float(row["ask"])
            mid = float(row["mid"])
            last = float(row.get("lastPrice") or row.get("last") or 0)

            # Hesap için kullanılacak fiyat: önce mid, yoksa last
            price_for_iv = mid if mid > 0 else last
            if price_for_iv <= 0:
                continue

            g = compute_greeks(
                option_type=opt_type, S=underlying, K=strike,
                T=T, r=risk_free_rate, market_price=price_for_iv,
            )
            if g is None:
                continue

            # Madde 16 — şişmiş IV guard
            if g.iv > cfg.skip_if_iv_above:
                continue

            # Madde 27 — delta tolerans (hedef ± tolerance)
            if abs(abs(g.delta) - target_delta) > cfg.delta_tolerance + 0.05:
                # Tolerans + küçük buffer — tamamen uzaktakileri at
                continue

            score = _score_contract(
                target_delta=target_delta, contract_delta=g.delta,
                open_interest=float(row["openInterest"] or 0),
                spread_pct=float(row["spread_pct"]),
            )
            if score > best_score:
                best_score = score
                best = OptionPick(
                    symbol=symbol, option_type=opt_type, strike=strike, expiry=expiry,
                    bid=bid, ask=ask, mid=mid, last_price=last,
                    open_interest=int(row["openInterest"] or 0),
                    volume=int(row["volume"] or 0),
                    spread_pct=float(row["spread_pct"]),
                    greeks=g, iv=g.iv, dte=dte,
                    underlying_price=underlying,
                    settlement_info=settlement,
                    reasons=[
                        f"DTE_{dte}",
                        f"STRIKE_{strike:.1f}",
                        f"DELTA_{g.delta:+.2f}",
                        f"IV_{g.iv*100:.0f}%",
                        f"OI_{int(row['openInterest'] or 0)}",
                        f"SPREAD_{float(row['spread_pct'])*100:.1f}%",
                        f"SCORE_{score:.2f}",
                    ],
                )

    if best is None:
        reasons = diag_reasons or ["NO_VIABLE_CONTRACT"]
        return None, reasons

    # Madde 25 (Tur-58) — ex-temettü erken-kullanım eşiği. Seçilen kontrat üzerinde
    # ölçülür: eşik prim (ekstrinsik) ister, o da ancak kontrat SEÇİLDİKTEN sonra bilinir.
    # Varsayılan İŞARETLER, bloke ETMEZ — SSR/borç bayrağı sözleşmesiyle aynı (Tur-57):
    # `EXDIV_GUARD=block` bilinçli olarak kapıya çevirir.
    try:
        days_div, div_amt = ex_dividend_context(symbol)
        ee = early_exercise_risk(
            option_type=best.option_type, strike=best.strike, underlying=best.underlying_price,
            option_mid=best.mid or best.last_price, days_to_ex_div=days_div,
            dividend=div_amt, symbol=symbol,
        )
        best.early_exercise = ee
        if ee["risk"]:
            tag = "EXDIV_RISK" if ee["measured"] else "EXDIV_RISK_UNMEASURED"
            best.reasons.append(f"{tag}_{ee['reason']}")
            if (os.getenv("EXDIV_GUARD") or "").strip().lower() == "block":
                return None, [f"EXDIV_BLOCKED_{ee['reason']}"]
            logger.info(f"[opt] {symbol} ex-temettü erken-kullanım eşiği: {ee['reason']} "
                        f"(ekstrinsik={ee['extrinsic']}, D={ee['dividend']}, gün={days_div})")
    except Exception as e:      # noqa: BLE001 — takvim yolu sinyali düşürmez
        logger.debug(f"[opt] {symbol} ex-div eşiği ölçülemedi: {e}")

    # Madde 16 — ATM yakını IV'yi günlük cache'e ekle (52w rolling icin)
    # (Seçilen kontrat hedef delta = 0.50 ise yaklaşık ATM'dir.)
    try:
        append_iv_today(symbol, best.iv)
        ivr_result = get_iv_rank(symbol, best.iv)
        best.reasons.append(f"IVR_{ivr_result.iv_rank:.0f}_{ivr_result.confidence}")
    except Exception as e:
        logger.debug(f"[opt] IV history kaydi basarisiz {symbol}: {e}")

    return best, list(best.reasons)


def get_option_premium_now(
    provider: DataProvider, symbol: str, option_type: str,
    strike: float, expiry: str,
) -> float | None:
    """Açık opsiyon pozisyonu için anlık premium (mid). Bulunamazsa None."""
    chain = provider.get_option_chain(symbol, expiry)
    if not chain:
        return None
    df = chain.get("calls" if option_type == "c" else "puts")
    if df is None or df.empty:
        return None
    match = df[np.isclose(df["strike"], strike, atol=0.01)]
    if match.empty:
        return None
    row = match.iloc[0]
    bid = float(row["bid"] or 0)
    ask = float(row["ask"] or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    last = float(row.get("lastPrice") or row.get("last") or 0)
    return last if last > 0 else None
