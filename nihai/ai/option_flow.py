"""
option_flow.py — OPSİYON AÇIK POZİSYON AKIŞI ve PİYASA YAPICI KONUMU (Tur-57). KARAR-DIŞI.

`options_math.py` TEK bir kontratın/gülümsemenin matematiğini taşır (IV, greekler, SVI,
RND). Eksik olan aile şuydu: **zincirin TAMAMI** üstünde AÇIK POZİSYON (OI) ve HACİM
ağırlıklı toplamlar — yani piyasa yapıcının hedge etmek ZORUNDA olduğu miktarın haritası:

  - `dealer_gamma_exposure` : strike başına $GEX + net GEX + **gamma flip** seviyesi +
                              çağrı/satım duvarları (hedge yönünün döndüğü fiyat)
  - `put_call_ratio`        : OI-PCR ve hacim-PCR (korunma/panik yoğunluğu)
  - `max_pain`              : vade gününde opsiyon YAZARININ ödemesini en aza indiren spot
  - `unusual_activity`      : hacim > OI olan kontratlar (yeni pozisyon açılışı imzası)
  - `dealer_vanna_charm`    : vol düşerken (vanna) ve zaman akarken (charm) doğan zorunlu akış

ANAYASA: KARAR-DIŞI. Buradaki hiçbir çıktı sinyale/eşiğe/META_GATE'e/execution'a bağlanmaz.
"Net GEX negatif" bir kenar İDDİASI DEĞİLDİR (ACI DERS #3) — bu modül TEŞHİS aracıdır; bir
aile aday olacaksa `research/edge_lab` kanıt kapısından geçmek ZORUNDADIR ve kenar-keşif
defterinin `n_trials` sayacını artırır (Tur-55 dersi).

ÖLÇÜLMEMİŞ VARSAYIM (dürüstlük notu): dealer'ın call'da UZUN, put'ta KISA olduğu klasik
kabul edilir (GEX literatürünün standardı). Bu kabul işlem TARAFI verisi olmadan
DOĞRULANAMAZ; yanlışsa net GEX'in İŞARETİ ters döner. Bu yüzden fonksiyonlar
`dealer_convention` alanını çıktıya yazar — sayı, varsayımından ayrılmaz.

ÖLÇÜM YOKLUĞU ÖLÇÜM DEĞİLDİR: OI kolonu olmayan zincirde (DoltHub önbelleği — Tur-54
bulgusu) ya da hacim kolonu olmayan zincirde (Alpaca OPRA anlık zinciri) ilgili fonksiyon
None döner veya `measured=False` basar; sıfırla DOLDURMAZ.

Savunmacı sözleşme `quant_math`/`options_math` ile aynı: kısa/NaN/dönüştürülemez girdi →
None; hiçbir girdi istisna fırlatmaz. Saf fonksiyonlar; numpy dışında bağımlılık yok.

Kaynaklar: Black-Scholes (1973) gamma/vanna/charm kapalı formları; "gamma exposure" ve
"gamma flip" kavramlarının piyasa kullanımı (SqueezeMetrics 2017); max pain'in klasik
yazar-ödemesi tanımı.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

# Bir opsiyon kontratı kaç dayanak hisseyi temsil eder (ABD borsa standardı).
CONTRACT_SIZE = 100

_SQRT_2PI = math.sqrt(2.0 * math.pi)

# yfinance zincir kolon adları → bu modülün kanonik alanları. Alpaca/DoltHub yolları da
# aynı şemaya normalize edilir (nihai/data/alpaca_provider.py), böylece TEK adaptör yeter.
_COLS: dict[str, tuple[str, ...]] = {
    "strike": ("strike", "strike_price"),
    "oi": ("openInterest", "open_interest", "oi"),
    "volume": ("volume", "vol"),
    "iv": ("impliedVolatility", "implied_volatility", "iv"),
    "gamma": ("gamma",),
}


def _pick(obj: object, names: tuple[str, ...]) -> np.ndarray | None:
    """Zincir nesnesinden (DataFrame VEYA dict) ilk eşleşen kolonu float diziye çevir."""
    for n in names:
        try:
            col = obj[n]  # type: ignore[index]
        except Exception:
            continue
        try:
            arr = np.asarray(col, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if arr.size:
            return arr
    return None


def chain_side(obj: object) -> dict[str, np.ndarray] | None:
    """Zincirin bir tarafını (calls VEYA puts) kanonik alanlara çevir — SAF adaptör.

    Girdi: pandas DataFrame (yfinance şeması) veya {kolon: dizi} sözlüğü. Çıktı yalnız
    BULUNAN alanları taşır; `strike` yoksa None. Eksik alan UYDURULMAZ — sıfırla doldurmak
    "OI yok"u "OI sıfır" gibi gösterirdi."""
    if obj is None:
        return None
    strike = _pick(obj, _COLS["strike"])
    if strike is None or strike.size == 0:
        return None
    out: dict[str, np.ndarray] = {"strike": strike}
    for key, names in _COLS.items():
        if key == "strike":
            continue
        arr = _pick(obj, names)
        if arr is not None and arr.size == strike.size:
            out[key] = arr
    return out


def _valid(side: dict[str, np.ndarray] | None, field: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(strike, alan) çiftini sonlu ve pozitif-strike satırlarla sınırla."""
    if not side or field not in side:
        return None
    k = np.asarray(side["strike"], dtype=float)
    v = np.asarray(side[field], dtype=float)
    m = np.isfinite(k) & np.isfinite(v) & (k > 0)
    if not np.any(m):
        return None
    return k[m], v[m]


def bs_gamma_vec(s: float, strikes: ArrayLike, t: float, r: float,
                 sigma: ArrayLike) -> np.ndarray | None:
    """Vektörel Black-Scholes gamma — `quant_math.bs_greeks`'in dizi hâli.

    Skaler eşdeğerlik `tests/test_tur57_akis_ve_veri.py`'de kilitlidir: formül iki yerde
    yaşıyor, parite testi olmadan biri sessizce kayar."""
    try:
        sf, tf, rf = float(s), float(t), float(r)
    except (TypeError, ValueError):
        return None
    k = np.asarray(strikes, dtype=float).reshape(-1)
    sig = np.asarray(sigma, dtype=float).reshape(-1)
    if sig.size == 1 and k.size > 1:
        sig = np.repeat(sig, k.size)
    if k.size != sig.size or k.size == 0 or not math.isfinite(sf) or sf <= 0 or tf <= 0:
        return None
    out = np.full(k.size, np.nan)
    ok = np.isfinite(k) & np.isfinite(sig) & (k > 0) & (sig > 0)
    if not np.any(ok):
        return out
    sqt = math.sqrt(tf)
    d1 = (np.log(sf / k[ok]) + (rf + 0.5 * sig[ok] ** 2) * tf) / (sig[ok] * sqt)
    out[ok] = np.exp(-0.5 * d1 * d1) / _SQRT_2PI / (sf * sig[ok] * sqt)
    return out


class _Side:
    """Bir tarafın (call/put) GEX için gereken sadeleştirilmiş görünümü."""

    __slots__ = ("strike", "oi", "gamma", "iv", "source")

    def __init__(self, strike: np.ndarray, oi: np.ndarray, gamma: np.ndarray,
                 iv: np.ndarray | None, source: str) -> None:
        self.strike, self.oi, self.gamma, self.iv, self.source = strike, oi, gamma, iv, source

    def gamma_at(self, spot: float, t: float, r: float) -> np.ndarray:
        """Verilen spot'ta gamma. IV biliniyorsa YENİDEN fiyatlanır (gamma spot'a bağlıdır);
        yalnız hazır gamma varsa o sabit kabul edilir (kaynağın söyleyebildiği kadarı)."""
        if self.iv is None:
            return self.gamma
        g = bs_gamma_vec(spot, self.strike, t, r, self.iv)
        return self.gamma if g is None else np.where(np.isfinite(g), g, self.gamma)


def _prepare_side(obj: object, spot: float, t: float, r: float) -> _Side | None:
    """Zincir tarafını GEX görünümüne çevir: OI zorunlu, gamma zincirden ya da IV'den."""
    side = chain_side(obj)
    if not side or "oi" not in side:
        return None
    k = np.asarray(side["strike"], dtype=float)
    oi = np.asarray(side["oi"], dtype=float)
    m = np.isfinite(k) & np.isfinite(oi) & (k > 0)
    if not np.any(m):
        return None
    k, oi = k[m], oi[m]
    iv = np.asarray(side["iv"], dtype=float)[m] if "iv" in side else None
    g = bs_gamma_vec(spot, k, t, r, iv) if iv is not None else None
    if g is None:
        g = np.full(k.size, np.nan)
    g2 = np.asarray(side["gamma"], dtype=float)[m] if "gamma" in side else np.full(k.size, np.nan)
    zincir = ~np.isfinite(g) & np.isfinite(g2) & (g2 >= 0)
    source = "karisik" if np.any(np.isfinite(g)) and np.any(zincir) else (
        "iv_turetildi" if np.any(np.isfinite(g)) else "zincir")
    # Eksik gamma NaN kalır: toplama girmez, kapsam sayımından kaybolmaz.
    return _Side(k, oi, np.where(zincir, g2, g), iv, source)


def dealer_gamma_exposure(calls: object, puts: object, spot: float, t: float,
                          r: float = 0.0, contract_size: int = CONTRACT_SIZE,
                          grid: int = 61, span: float = 0.15) -> dict | None:
    """Strike başına $GEX, net GEX, **gamma flip** seviyesi ve duvarlar.

    $GEX(K) = Γ(K) · OI(K) · contract_size · S² · 0.01 → dayanağın %1'lik hareketinde
    piyasa yapıcının hedge için almak/satmak zorunda kalacağı DOLAR tutarı.

    Net GEX = Σ_call − Σ_put (dealer call'da uzun, put'ta kısa varsayımı; modül
    docstring'indeki uyarı geçerlidir). Pozitif → hedge akışı hareketi SÖNÜMLER (vol
    düşer, bant içi), negatif → hedge akışı hareketi BÜYÜTÜR (vol patlar, trend).

    **Gamma flip** kaba bir eşik değil ÖLÇÜLMÜŞ bir köktür: net GEX spot ızgarasında
    (S·(1±span), `grid` nokta) gammalar YENİDEN fiyatlanarak hesaplanır, işaret
    değiştiren aralıkta brentq ile kök çözülür. Strike üstünde kümülatif toplam almak
    (yaygın kısayol) gamma'nın spot bağımlılığını yok sayar ve yanlış seviye üretir.

    Verilen bir tarafta geçerli strike var ama sonlu OI hiç yoksa → None; hiç
    verilmeyen taraf (None) nete katılmaz. Kısmi OI/gamma eksikleri toplama girmez,
    `n_oi_eksik`/`n_gamma_eksik` ile sayılır. IV geçerliyse gamma yeniden fiyatlanır,
    değilse ölçülmüş zincir gamması kullanılır. OI taşıyan tüm satırlar IV ile
    fiyatlanamıyorsa gamma flip ölçülemez. Verilen bir tarafta hiç gamma/IV ölçümü yoksa → None."""
    try:
        sf, tf, rf = float(spot), float(t), float(r)
        cs, grid, span = int(contract_size), int(grid), float(span)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (all(math.isfinite(x) for x in (sf, tf, rf, span)) and sf > 0 and tf > 0
            and cs > 0 and 0 < span < 1):
        return None
    n_oi_eksik = 0
    for obj in (calls, puts):
        side = chain_side(obj)
        if side is None:
            continue
        valid_k = np.isfinite(side["strike"]) & (side["strike"] > 0)
        oi_ok = np.isfinite(side["oi"]) if "oi" in side else np.zeros(valid_k.size, dtype=bool)
        if np.any(valid_k) and not np.any(valid_k & oi_ok):
            return None
        n_oi_eksik += int(np.count_nonzero(valid_k & ~oi_ok))
    c = _prepare_side(calls, sf, tf, rf)
    p = _prepare_side(puts, sf, tf, rf)
    # OI'si ölçülü ama hiç gamma/IV taşımayan taraf da o tarafı nete SIFIR yazardı (O1 ile aynı
    # işaret hatası; Tur-78 incelemesi): kısmi eksik sayılır, TAMAMEN eksik taraf → None.
    if (c is None and p is None) or any(s is not None and not np.any(np.isfinite(s.gamma)) for s in (c, p)):
        return None

    scale = cs * 0.01

    def _by_strike(side: _Side | None, sign: float, ks: np.ndarray) -> np.ndarray:
        out = np.zeros(ks.size)
        if side is None:
            return out
        val = sign * side.gamma * side.oi * scale * sf * sf
        for i, k in enumerate(ks):
            sel = side.strike == k
            if np.any(sel):
                out[i] = float(np.nansum(val[sel]))
        return out

    parts = [s.strike for s in (c, p) if s is not None]
    strikes = np.unique(np.concatenate(parts))
    per_call = _by_strike(c, +1.0, strikes)
    per_put = _by_strike(p, -1.0, strikes)
    net_curve = per_call + per_put

    def _net_at(s_try: float) -> float:
        tot = 0.0
        for side, sign in ((c, +1.0), (p, -1.0)):
            if side is None:
                continue
            g = side.gamma_at(s_try, tf, rf)
            tot += sign * float(np.nansum(np.nan_to_num(g) * side.oi)) * scale * s_try * s_try
        return tot

    lo, hi = sf * (1.0 - abs(span)), sf * (1.0 + abs(span))
    grid_s = np.linspace(lo, hi, max(11, min(int(grid), 401)))
    grid_net = np.array([_net_at(float(x)) for x in grid_s])
    roots = []
    if not np.all(np.isfinite(grid_net)):
        return None
    # Sabit gamma'yı spot boyunca taşımak bir gamma-flip ÖLÇÜMÜ değildir.
    if all(side is None or (side.iv is not None and np.all(np.isfinite(side.iv) & (side.iv > 0)))
           for side in (c, p)) and np.any(grid_net != 0):
        cross = np.where(np.signbit(grid_net[:-1]) != np.signbit(grid_net[1:]))[0]
        for i in cross:
            try:
                roots.append(float(brentq(_net_at, float(grid_s[i]), float(grid_s[i + 1]), xtol=1e-10)))
            except (ValueError, RuntimeError):
                continue
    flip = min(roots, key=lambda value: abs(value - sf)) if roots else None

    net_now = float(np.nansum(net_curve))
    sources = {s.source for s in (c, p) if s is not None and np.any(np.isfinite(s.gamma))}
    return {
        "strikes": strikes,
        "call_gex": per_call,
        "put_gex": per_put,
        "net_gex_by_strike": net_curve,
        "net_gex": net_now,
        "net_gex_full_move": net_now / 0.01,
        "gex_unit": "USD_per_1pct_spot_move",
        "net_gex_bn": net_now / 1e9,
        "gamma_flip": flip,
        "spot_vs_flip": (None if flip is None else float(sf - flip)),
        "regime": "pozitif_gamma" if net_now > 0 else ("negatif_gamma" if net_now < 0 else "notr_gamma"),
        "call_wall": float(strikes[int(np.argmax(per_call))]) if np.any(per_call > 0) else None,
        "put_wall": float(strikes[int(np.argmin(per_put))]) if np.any(per_put < 0) else None,
        "grid_spot": grid_s,
        "grid_net_gex": grid_net,
        "gamma_source": next(iter(sources)) if len(sources) == 1 else "karisik",
        "n_oi_eksik": n_oi_eksik,
        "n_gamma_eksik": sum(int(np.count_nonzero(~np.isfinite(s.gamma))) for s in (c, p) if s is not None),
        "dealer_convention": "call_long_put_short",
        "spot": sf,
        "n_strikes": int(strikes.size),
    }


def put_call_ratio(calls: object, puts: object) -> dict | None:
    """OI ve HACİM put/call oranları. Bir taraf ölçülmemişse o oran None olur (0 DEĞİL)."""
    c, p = chain_side(calls), chain_side(puts)
    if c is None and p is None:
        return None
    out: dict = {"oi_pcr": None, "volume_pcr": None, "measured": {}}
    for field in ("oi", "volume"):
        cv, pv = _valid(c, field), _valid(p, field)
        if cv is None or pv is None:
            out["measured"][field] = False
            continue
        cs, ps = float(np.nansum(cv[1])), float(np.nansum(pv[1]))
        out[f"{field}_call"], out[f"{field}_put"] = cs, ps
        out["measured"][field] = True
        out["oi_pcr" if field == "oi" else "volume_pcr"] = (ps / cs) if cs > 0 else None
    if out["oi_pcr"] is None and out["volume_pcr"] is None:
        return None
    return out


def max_pain(calls: object, puts: object) -> dict | None:
    """Max pain: vade gününde opsiyon YAZARININ toplam ödemesini en aza indiren spot.

    Ödeme(S*) = Σ_call OI·max(S*−K,0) + Σ_put OI·max(K−S*,0). Aday S* kümesi zincirdeki
    strike'lardır: ödeme parçalı-doğrusal ve dışbükeydir, kırılımlar tam da
    strike'lardadır — arada bir minimum OLAMAZ. OI yoksa → None."""
    c = _valid(chain_side(calls), "oi")
    p = _valid(chain_side(puts), "oi")
    if c is None and p is None:
        return None
    ks = np.unique(np.concatenate([x[0] for x in (c, p) if x is not None]))
    if ks.size == 0:
        return None
    pain = np.zeros(ks.size)
    for i, s in enumerate(ks):
        tot = 0.0
        if c is not None:
            tot += float(np.nansum(np.maximum(s - c[0], 0.0) * c[1]))
        if p is not None:
            tot += float(np.nansum(np.maximum(p[0] - s, 0.0) * p[1]))
        pain[i] = tot
    j = int(np.argmin(pain))
    return {"max_pain": float(ks[j]), "strikes": ks, "pain": pain,
            "pain_min": float(pain[j]), "pain_max": float(np.max(pain)),
            "n_strikes": int(ks.size)}


def unusual_activity(calls: object, puts: object, min_volume: float = 50.0,
                     ratio: float = 1.0, top: int = 15) -> dict | None:
    """UOA — günlük hacmi açık pozisyonunu AŞAN kontratlar (yeni pozisyon açılışı imzası).

    Hacim VE OI kolonu BİRLİKTE gerekir; yalnız biri varsa oran hesaplanamaz → None
    (Alpaca OPRA anlık zincirinde hacim yoktur, yfinance zincirinde ikisi de vardır).
    `ratio` vol/OI eşiği, `min_volume` gürültü tabanı."""
    rows: list[dict] = []
    olculebilir = False
    for label, obj in (("call", calls), ("put", puts)):
        side = chain_side(obj)
        if not side or "volume" not in side or "oi" not in side:
            continue
        olculebilir = True
        k, v, o = side["strike"], side["volume"], side["oi"]
        m = np.isfinite(k) & np.isfinite(v) & np.isfinite(o) & (k > 0) & (v >= float(min_volume))
        with np.errstate(divide="ignore", invalid="ignore"):
            rt = np.where(o > 0, v / np.where(o > 0, o, 1.0), np.inf)
        for i in np.where(m & (rt >= float(ratio)))[0]:
            rows.append({"kind": label, "strike": float(k[i]), "volume": float(v[i]),
                         "oi": float(o[i]),
                         "vol_oi": float(rt[i]) if np.isfinite(rt[i]) else None})
    if not olculebilir:
        return None
    rows.sort(key=lambda d: (d["vol_oi"] is None, -(d["vol_oi"] or 0.0)))
    return {"rows": rows[: max(1, int(top))], "n_hit": len(rows),
            "min_volume": float(min_volume), "ratio": float(ratio)}


def dealer_vanna_charm(calls: object, puts: object, spot: float, t: float,
                       r: float = 0.0, contract_size: int = CONTRACT_SIZE) -> dict | None:
    """Zincir toplamı vanna ve charm maruziyeti — zorunlu akışın iki ikincil kaynağı.

    vanna = ∂²V/∂S∂σ = −φ(d1)·d2/σ → vol değiştikçe dealer'ın spot hedge'i kayar.
    charm = ∂²V/∂S∂t               → zaman aktıkça delta kayar (vade haftası akışı).

    Formüller `options_math.bs_greeks_extended` ile AYNIDIR (orada skaler, burada dizi);
    parite testi ikisini kilitler. Hiç ölçülebilir satır yoksa ya da verilen bir tarafın
    geçerli strike'ları hiç ölçülemiyorsa → None. Kısmi olarak IV/OI'si eksik satırlar
    `n_eksik` ile sayılır, toplama girmez."""
    try:
        sf, tf, rf = float(spot), float(t), float(r)
        cs = int(contract_size)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (all(math.isfinite(x) for x in (sf, tf, rf)) and sf > 0 and tf > 0 and cs > 0):
        return None
    tot_vanna = tot_charm = 0.0
    used = n_eksik = 0
    for obj, sign in ((calls, +1.0), (puts, -1.0)):
        side = chain_side(obj)
        if not side:
            continue
        k = np.asarray(side["strike"], dtype=float)
        valid_k = np.isfinite(k) & (k > 0)
        if "iv" not in side or "oi" not in side:
            m = np.zeros(k.size, dtype=bool)
        else:
            iv = np.asarray(side["iv"], dtype=float)
            oi = np.asarray(side["oi"], dtype=float)
            m = np.isfinite(k) & np.isfinite(iv) & np.isfinite(oi) & (k > 0) & (iv > 0) & (oi >= 0)
        n_eksik += int(np.count_nonzero(valid_k & ~m))
        if not np.any(m):
            if np.any(valid_k):       # verilmiş ama hiç ölçülemeyen taraf nete sıfır yazılamaz
                return None
            continue
        sqt = math.sqrt(tf)
        d1 = (np.log(sf / k[m]) + (rf + 0.5 * iv[m] ** 2) * tf) / (iv[m] * sqt)
        d2 = d1 - iv[m] * sqt
        pdf = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
        vanna = -pdf * d2 / iv[m]
        charm = -pdf * (2.0 * rf * tf - d2 * iv[m] * sqt) / (2.0 * tf * iv[m] * sqt)
        tot_vanna += sign * float(np.nansum(vanna * oi[m])) * cs
        tot_charm += sign * float(np.nansum(charm * oi[m])) * cs
        used += int(np.count_nonzero(m))
    if used == 0:
        return None
    return {"vanna_exposure": tot_vanna, "charm_exposure": tot_charm,
            "vanna_per_vol_pt": tot_vanna * 0.01 * sf,
            "charm_per_day": tot_charm / 365.0 * sf,
            "n_contracts": used, "n_eksik": n_eksik, "dealer_convention": "call_long_put_short"}


def dealer_hedging_clock(calls: object, puts: object, spot: float, t: float, r: float = 0.0, *,
                         hours: int = 6, iv_change_per_hour: float = 0.0) -> dict | None:
    """Saatlik vanna/charm hedge senaryosu; pozitif USD dealer dayanak ALIMI'dır.

    Spot/OI sabit, IV değişimi çağıranın senaryosudur. Her saatin başında Greeks
    yeniden ölçülür. Vade sonrası satır yok; eksik IV/OI ile akış uydurulmaz.
    """
    try:
        sf, tf, rf, change = float(spot), float(t), float(r), float(iv_change_per_hour)
        if int(hours) != hours or not 1 <= hours <= 24:
            return None
        if not all(math.isfinite(x) for x in (sf, tf, rf, change)) or sf <= 0 or tf <= 0:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    sides = [chain_side(side) for side in (calls, puts)]
    if any(side is not None and ("iv" not in side or "oi" not in side) for side in sides):
        return None
    for side in sides:
        if side is not None:
            valid_k = np.isfinite(side["strike"]) & (side["strike"] > 0)
            if not np.all(np.isfinite(side["oi"][valid_k])):
                return None
    rows = []
    cumulative = 0.0
    for hour in range(int(hours)):
        left = tf * 365 * 24 - hour
        if left <= 1e-10:
            break
        step = min(1.0, left)
        shifted: list[dict[str, np.ndarray] | None] = []
        for side in sides:
            if side is None:
                shifted.append(None)
                continue
            iv = side["iv"] + change * hour
            if not np.all(np.isfinite(iv)) or np.any(iv <= 0) or np.any(iv + change * step <= 0):
                return None
            shifted.append({**side, "iv": iv})
        exposures = dealer_vanna_charm(shifted[0], shifted[1], sf, left / (365 * 24), rf)
        if exposures is None:
            return None
        # ponytail: saat başına birinci dereceden yaklaşım; büyük IV şokunda tam
        # delta yeniden fiyatlaması gerekir. Hedge değişimi opsiyon deltasının TERSİ.
        vanna = -exposures["vanna_exposure"] * change * step * sf
        charm = -exposures["charm_exposure"] * step / (365 * 24) * sf
        cumulative += vanna + charm
        if not all(math.isfinite(x) for x in (vanna, charm, cumulative)):
            return None
        rows.append({"hour": hour + step, "vanna_usd": vanna, "charm_usd": charm,
                     "net_usd": vanna + charm, "cumulative_usd": cumulative})
    return {"rows": rows, "iv_change_per_hour": change, "spot": sf,
            "dealer_convention": "call_long_put_short", "method": "hourly_first_order_fixed_spot"} if rows else None
