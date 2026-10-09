"""
options_analyzer.py — Opsiyon zinciri analizi, Greekler, IV rank, likidite filtresi.

PDF Referansları:
- Madde 21 : Opsiyon likidite ve zincir sağlığı (OI, hacim, makas, DTE)
- Madde 27 : Portföy Greek limitleri (net Delta/Theta/Vega/Gamma)
- Madde 28 : Vade, atama, pin riski; 0DTE
- Madde 29 : Uzlaşma stili (Amerikan/Avrupa, fiziki/nakit, endeks/1256)
- Madde 16 : IV Rank/Percentile — yüksek/düşük IV rejimi
- Madde 17 : IV Crush — şişmiş IV ile uzun opsiyon alma
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

# Tur-80: vollib'in greek modülü `scipy.stats`'ı çeker (import ≈1,3 sn) ve bu modülü import eden
# HER süreç — tarayıcı, pano, backtest, replay alt süreci — Greek hesaplamasa da o vergiyi
# ödüyordu. Varlık ucuzca sorulur; asıl import ilk Greek hesabında BİR KEZ yapılır ve başarısız
# olursa eskisi gibi vollib'siz dala düşülür.
_VOLLIB_BULUNDU = importlib.util.find_spec("vollib") is not None
_VOLLIB_FNS: tuple | None = None
if not _VOLLIB_BULUNDU:
    logger.warning("vollib bulunamadı — Greekler basit yaklaşımla hesaplanacak.")


def _vollib_fns() -> tuple | None:
    """(delta, gamma, theta, vega, implied_volatility) — ilk çağrıda import; yoksa None."""
    global _VOLLIB_FNS, _VOLLIB_BULUNDU
    if _VOLLIB_FNS is None and _VOLLIB_BULUNDU:
        try:
            from vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
            from vollib.black_scholes.implied_volatility import implied_volatility
            _VOLLIB_FNS = (delta, gamma, theta, vega, implied_volatility)
        except Exception:
            _VOLLIB_BULUNDU = False
            logger.warning("vollib bulunamadı — Greekler basit yaklaşımla hesaplanacak.")
    return _VOLLIB_FNS


def __getattr__(name: str):
    """`HAS_VOLLIB` eski anlamını korur: "vollib KULLANILABİLİR mi". Okunduğu an import denenir →
    paket bulunup import EDİLEMİYORSA False + uyarı (Codex incelemesi: tembel sürümde ilk Greek
    çağrısına kadar True kalıyordu). Bayrağı okumayan süreç importu hiç ödemez (PEP 562)."""
    if name == "HAS_VOLLIB":
        return _vollib_fns() is not None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


OptionType = Literal["c", "p"]

# Madde 29 — Endeks/ETF/Hisse meta tablo (basitleştirilmiş)
# Gerçek ürün listesi için OCC + broker dokümantasyonuna bakılır.
EUROPEAN_CASH_SETTLED_INDEX = {"SPX", "SPXW", "NDX", "VIX", "RUT", "XSP"}
SECTION_1256 = EUROPEAN_CASH_SETTLED_INDEX  # ABD: endeks opsiyonları 60/40


def settlement_style(symbol: str) -> dict:
    """Madde 29 — Kullanım stili + uzlaşma + vergi rejimi."""
    sym = symbol.upper()
    if sym in EUROPEAN_CASH_SETTLED_INDEX:
        return {"exercise": "European", "settlement": "Cash", "tax_1256": True}
    return {"exercise": "American", "settlement": "Physical", "tax_1256": False}


# ---------------------------------------------------------------------------
# Greekler
# ---------------------------------------------------------------------------
@dataclass
class GreekSet:
    delta: float
    gamma: float
    theta: float   # günlük (theta/365)
    vega: float    # %1 IV değişimi için
    iv: float


def _greeks_without_vollib(
    *, option_type: OptionType, S: float, K: float, T: float, r: float,
    market_price: float | None, sigma: float | None,
) -> GreekSet | None:
    """vollib YOKKEN Greekler — `nihai/ai` kendi Black-Scholes çekirdeğiyle (Tur-37).

    ÖNCE NE VARDI: `delta=±0.5, gamma=0.02, theta=-0.05, vega=0.10` — UYDURMA SABİTLER.
    Bu sayılar hiçbir S/K/T/σ'ya bakmıyordu; ATM 30 günlük bir kontratla 6 aylık derin OTM
    bir kontrat AYNI Greekleri alıyordu ve Madde 27 portföy Greek limitleri o sabitlerin
    üstünde işliyordu. Yani ölçüm YOKLUĞU bir ölçüm gibi raporlanıyordu (Tur-31'in IV Rank
    dersinin birebir tekrarı) — üstelik burada KARAR YOLUNDA.

    ŞİMDİ: `nihai/ai/options_math.implied_vol` (Brent kök bulma; Newton'un aksine derin
    OTM'de vega≈0 iken ıraksamaz) + `nihai/ai/quant_math.bs_greeks` (kapalı-form BS).
    İkisi de SAF MATEMATİKTİR — fit edilmiş parametresi, öğrenilmiş kenar iddiası, kanıt
    kapısı gerektiren bir modeli YOKTUR. Anayasanın koruduğu şey kanıtlanmamış MODELİN
    karara girmesidir; determinist bir formülün uydurma sabitin yerini alması o kapıyla
    ilgili değildir, tam tersine kapının ölçtüğü sayıyı dürüstleştirir.

    BİRİM HİZASI (ölçüldü, 6 ondalığa kadar birebir): `quant_math.bs_greeks` theta'yı YILLIK,
    vega'yı 1,0 vol birimi başına verir; `GreekSet` sözleşmesi (= vollib) theta'yı GÜNLÜK,
    vega'yı %1 IV başına ister → `/365` ve `/100`. Bu dönüşüm atlanırsa vollib'li ve
    vollib'siz yollar 365 ve 100 kat ayrışır — sessiz ve ölümcül bir tutarsızlık olurdu.

    SON ÇARE: AI katmanı da çözemezse (import yok / IV kökü yok / girdi dejenere) `None`
    döner. Uydurma sabitlere GERİ DÖNÜLMEZ: `compute_greeks`'in vollib'li dalı da
    çözemediğinde zaten None döndürüyor ve çağıranlar bunu bekliyor (`estimate_atm_iv`
    `if g and 0 < g.iv < 5.0` ile süzüyor). Yanlış sayı, sayı yokluğundan kötüdür."""
    try:
        from nihai.ai.options_math import implied_vol
        from nihai.ai.quant_math import bs_greeks
    except Exception as e:      # noqa: BLE001 — opsiyon yolu AI katmanına bağımlı olamaz
        logger.debug(f"[greeks] AI matematik katmanı yüklenemedi: {e}")
        return None

    if sigma is None:
        if market_price is None or market_price <= 0:
            return None
        sigma = implied_vol(market_price, S, K, T, r, option_type)
        if sigma is None:
            return None
    g = bs_greeks(S, K, T, r, sigma, option_type)
    if g is None:
        return None
    return GreekSet(delta=g["delta"], gamma=g["gamma"],
                    theta=g["theta"] / 365.0,     # yıllık → günlük (vollib sözleşmesi)
                    vega=g["vega"] / 100.0,       # 1,0 vol birimi → %1 (vollib sözleşmesi)
                    iv=float(sigma))


def compute_greeks(
    *, option_type: OptionType, S: float, K: float, T: float, r: float = 0.05,
    market_price: float | None = None, sigma: float | None = None,
) -> GreekSet | None:
    """
    Black-Scholes Greekler. T = yıl cinsinden vadeye kalan süre.
    market_price verilirse implied vol çözülür; aksi halde sigma gerekir.
    """
    if T <= 0 or S <= 0 or K <= 0:
        return None

    fns = _vollib_fns()
    if fns is None:
        return _greeks_without_vollib(option_type=option_type, S=S, K=K, T=T, r=r,
                                      market_price=market_price, sigma=sigma)
    delta, gamma, theta, vega, implied_volatility = fns

    try:
        if sigma is None:
            if market_price is None or market_price <= 0:
                return None
            sigma = implied_volatility(market_price, S, K, T, r, option_type)

        d = delta(option_type, S, K, T, r, sigma)
        g = gamma(option_type, S, K, T, r, sigma)
        th = theta(option_type, S, K, T, r, sigma)  # vollib günlük döner
        v = vega(option_type, S, K, T, r, sigma)
        return GreekSet(delta=d, gamma=g, theta=th, vega=v, iv=sigma)
    except Exception as e:
        logger.debug(f"[greeks] çözümlenemedi: {e}")
        return None


# ---------------------------------------------------------------------------
# IV Rank — Madde 16
# ---------------------------------------------------------------------------
def iv_rank(current_iv: float, iv_history: pd.Series) -> float:
    """IVR = (current - min) / (max - min) * 100. iv_history son 52 haftalık ATM IV."""
    if iv_history.empty:
        return 50.0
    lo, hi = iv_history.min(), iv_history.max()
    if hi - lo < 1e-9:
        return 50.0
    return float(np.clip((current_iv - lo) / (hi - lo) * 100, 0, 100))


def estimate_atm_iv(calls_df: pd.DataFrame, underlying: float, dte: int,
                    r: float = 0.05) -> float | None:
    """
    Madde 16 — Zincirden ATM'ye en yakın call ile implied vol tahmini.

    Canlı sinyal yolunda IV Rank beslemek için kullanılır (önceden iv_rank
    yalnızca opsiyon seçim yolunda hesaplanıyordu; hisse sinyaline None gidiyordu).
    """
    if calls_df is None or calls_df.empty or underlying <= 0 or dte <= 0:
        return None
    df = calls_df.copy()
    if "strike" not in df.columns:
        return None
    df["atm_dist"] = (df["strike"].astype(float) - underlying).abs()
    df = df.sort_values("atm_dist")
    T = dte / 365.0
    for _, row in df.head(5).iterrows():
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(row.get("lastPrice") or 0)
        if mid <= 0:
            continue
        g = compute_greeks(option_type="c", S=underlying, K=float(row["strike"]),
                           T=T, r=r, market_price=mid)
        if g and 0 < g.iv < 5.0:
            return g.iv
    return None


# ---------------------------------------------------------------------------
# Zincir filtresi — Madde 21
# ---------------------------------------------------------------------------
@dataclass
class ChainHealthResult:
    accepted: pd.DataFrame
    rejected_count: int
    reasons: dict[str, int]


def filter_liquid_chain(
    chain: pd.DataFrame,
    *,
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
) -> ChainHealthResult:
    """Madde 21 — ilikit strike'ları eler."""
    if chain.empty:
        return ChainHealthResult(chain, 0, {})

    df = chain.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]).clip(lower=0)
    df["spread_pct"] = np.where(df["mid"] > 0, df["spread"] / df["mid"], np.inf)

    reasons: dict[str, int] = {}
    mask_oi = df["openInterest"].fillna(0) >= min_oi
    mask_vol = df["volume"].fillna(0) >= min_volume
    mask_spread = df["spread_pct"] <= max_spread_pct
    mask_pos = df["bid"] > 0

    reasons["LOW_OI"] = int((~mask_oi).sum())
    reasons["LOW_VOL"] = int((~mask_vol).sum())
    reasons["WIDE_SPREAD"] = int((~mask_spread).sum())
    reasons["NO_BID"] = int((~mask_pos).sum())

    keep = mask_oi & mask_vol & mask_spread & mask_pos
    return ChainHealthResult(df[keep], int((~keep).sum()), reasons)


# ---------------------------------------------------------------------------
# Vade / DTE / Pin riski — Madde 28
# ---------------------------------------------------------------------------
def days_to_expiry(expiry: str) -> int:
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days


def is_pin_risk(strike: float, underlying: float, dte: int) -> bool:
    """Vade günü ATM'ye yakın ise pin riski. underlying DAYANAK fiyatı olmalı (prim değil)."""
    if underlying <= 0:
        return False
    return dte <= 1 and abs(strike - underlying) / underlying < 0.005


def assignment_risk_short_call(strike: float, underlying: float, days_to_ex_div: int | None) -> bool:
    """Madde 28 — ex-div öncesi ITM call'da erken atama riski (YAZAN tarafı).

    Bu sistem opsiyon YAZMAZ (tek bacak LONG call/put) → fonksiyonun canlı çağıranı
    yoktur ve bilinçle öyle bırakılmıştır: çok-bacaklı/kredi yapıları eklenirse
    kısa bacağın kapısı budur. LONG tarafın karşılığı `early_exercise_risk`'tir."""
    if days_to_ex_div is None:
        return False
    return underlying > strike and 0 <= days_to_ex_div <= 1


def early_exercise_risk(
    *,
    option_type: str,
    strike: float,
    underlying: float,
    option_mid: float | None,
    days_to_ex_div: int | None,
    dividend: float | None = None,
    symbol: str | None = None,
    window_days: int = 1,
) -> dict:
    """Madde 25 — ex-temettü öncesi ITM **LONG CALL**'ın erken-kullanım eşiği (Tur-58).

    `assignment_risk_short_call` yıllardır kodluydu ve **hiçbir üretim çağıranı yoktu**
    (yalnız testler ve OPTİMİZASYON_LOG); dahası bu sistemde uygulanamazdı, çünkü kısa
    bacak YOK. Uzun bacakta olay AYNIDIR ama işareti terstir:

        ex-tarihte dayanak ≈ D kadar düşer. Uzun call temettüyü ALMAZ; içsel değerin
        ≈D'sini kaybeder. Kayıp yalnız kontratı ex-tarihten ÖNCE KULLANARAK önlenir ve
        kullanmak zaman değerini (ekstrinsik) feda eder. Yani **ekstrinsik < D** olduğu
        anda pozisyon "erken kullan ya da temettüyü yitir" ikilemine girer.

    Bu bir tahmin DEĞİL bir aritmetiktir (ACI DERS #3 kapsamı dışıdır — kanıt kapısı
    kanıtlanmamış MODELİ karar yolundan uzak tutar, determinist bir kimliği değil).

    ÖLÇÜM YOKLUĞU sözleşmesi (Tur-31/34): temettü TUTARI bilinmiyorsa `measured=False`
    döner ve karar yalnız "ITM + ex-güne ≤ window_days" sezgiseline dayanır — çağıran
    ikisini AYIRT EDEBİLİR ve bu projede varsayılan davranış sezgiselde **işaretlemek**,
    ölçülmüşte de yine işaretlemektir (bloke etmek `EXDIV_GUARD=block` ile OPT-IN).

    Avrupa/nakit uzlaşılı (Madde 29 · Section 1256: SPX/NDX/VIX/RUT/XSP) kontratta erken
    kullanım YOKTUR → risk yapısı gereği False; sebep taşınır ki "ölçmedik" ile
    "olamaz" karışmasın.
    """
    out: dict = {"risk": False, "measured": dividend is not None and dividend > 0,
                 "extrinsic": None, "dividend": dividend, "days_to_ex_div": days_to_ex_div,
                 "reason": "NO_EX_DIV"}
    if symbol and settlement_style(symbol).get("exercise") == "European":
        out["reason"] = "EUROPEAN_NO_EARLY_EXERCISE"
        out["measured"] = True      # ÖLÇÜLMEDİ değil, YAPISAL olarak yok
        return out
    if (option_type or "").lower() != "c":
        out["reason"] = "PUT_NO_EX_DIV_RISK"        # temettü düşüşü put'un LEHİNEDİR
        return out
    if days_to_ex_div is None:
        out["reason"] = "EX_DIV_DATE_UNKNOWN"
        return out
    if not (0 <= days_to_ex_div <= window_days):
        out["reason"] = f"EX_DIV_IN_{days_to_ex_div}_DAYS"
        return out
    if underlying <= 0 or strike <= 0 or underlying <= strike:
        out["reason"] = "OTM_NO_EARLY_EXERCISE"     # ITM olmayan call kullanılmaz
        return out

    intrinsic = underlying - strike
    if option_mid is not None and option_mid > 0:
        out["extrinsic"] = max(option_mid - intrinsic, 0.0)
    if out["measured"] and out["extrinsic"] is not None:
        out["risk"] = out["extrinsic"] < float(dividend or 0.0)
        out["reason"] = ("EXDIV_EXTRINSIC_BELOW_DIV" if out["risk"]
                         else "EXDIV_EXTRINSIC_COVERS_DIV")
        return out
    # Tutar (veya prim) yok → yalnız sezgisel: ex-güne ≤ window ve ITM
    out["risk"] = True
    out["measured"] = False
    out["reason"] = "EXDIV_ITM_UNMEASURED"
    return out


# ---------------------------------------------------------------------------
# Portföy Greek toplama — Madde 27
# ---------------------------------------------------------------------------
@dataclass
class PortfolioGreeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    def add_position(self, g: GreekSet, contracts: int, multiplier: int = 100) -> None:
        # BUG düzeltmesi: vollib theta/vega de delta/gamma gibi HİSSE-BAŞI döner;
        # pozisyon değeri için ×100 (kontrat çarpanı) gerekir. Eski kod theta/vega'yı
        # ×100'süz topluyordu → net theta/vega 100× küçük → Madde 27 theta/vega
        # tavanları (örn. -50 $/gün, 300 vega) ASLA tetiklenmiyordu. Cap'ler pozisyon
        # ölçeğinde kalibre (delta 200 ≈ "200 hisse eşdeğeri") → hepsi 'scale' ile.
        scale = contracts * multiplier
        self.delta += g.delta * scale
        self.gamma += g.gamma * scale
        self.theta += g.theta * scale
        self.vega += g.vega * scale

    def within_caps(self, caps) -> tuple[bool, list[str]]:
        breaches = []
        if abs(self.delta) > caps.max_net_delta:
            breaches.append(f"NET_DELTA {self.delta:.0f} > {caps.max_net_delta}")
        if self.theta < caps.max_net_theta:
            breaches.append(f"NET_THETA {self.theta:.2f} < {caps.max_net_theta}")
        if abs(self.vega) > caps.max_net_vega:
            breaches.append(f"NET_VEGA {self.vega:.0f} > {caps.max_net_vega}")
        if abs(self.gamma) > caps.max_net_gamma:
            breaches.append(f"NET_GAMMA {self.gamma:.2f} > {caps.max_net_gamma}")
        return (len(breaches) == 0, breaches)
