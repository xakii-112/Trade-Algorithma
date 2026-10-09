"""
market_structure.py — Halt/LULD ve SSR (açığa satış kısıtı) sinyal-tarafı koruması.

PDF Referansları:
- Madde 24 : İşlem durdurma (Halt) ve LULD yönetimi
- Madde 30 : Açığa satış — SSR (Reg SHO Rule 201)

NOT: Bu sistem emir göndermez. Burada yapılan, broker'a değil SİNYALE yönelik
korumadır: olası halt/LULD'da yeni sinyal üretmemek ve SSR aktifken short
sinyalini "kısıtlı" olarak işaretlemek. Gerçek halt/borrow/locate verisi canlı
broker/borsa feed'i gerektirir; burada gecikmeli OHLCV üzerinden sezgisel tespit
yapılır.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from nihai.config import CONFIG, DATA_DIR

# ---------------------------------------------------------------------------
# HALKA AÇIK HİSSE ADEDİ (float) — BULGU-59-8
# ---------------------------------------------------------------------------
# NEDEN: `ai/microstructure.turnover_rate(volume, float_shares)` yazılıydı ama HİÇBİR
# çağıranı yoktu ve pano açıkça *"float kaynağı bağlı değil"* basıyordu — yani ölçüm
# yokluğu dürüstçe raporlanıyor ama KAPATILMIYORDU. Kaynak yeni bir bağımlılık gerektirmez:
# `yfinance` zaten kurulu ve `floatShares`/`sharesOutstanding` alanlarını verir.
#
# ÜÇ SINIR AÇIKÇA TAŞINIR (Tur-34 "ölçüm koşulu taşınmalıdır"):
#   1. Float **anlıktır, point-in-time DEĞİLDİR** — sağlayıcı bugünkü değeri döndürür.
#      Bu yüzden `float_shares` yalnız CANLI/pano yolunda kullanılmalıdır; tarihsel bir
#      dönüşüm hızı serisi kurmak look-ahead olurdu (`as_of` alanı YOK).
#   2. Ölçülemezse **None** döner — 0 ya da "sharesOutstanding'e düş" gibi bir uydurma
#      yapılmaz; `float` ile `outstanding` FARKLI büyüklüklerdir (kilitli/içeriden paylar).
#      İkincisi yalnız AÇIKÇA istendiğinde (`allow_outstanding=True`) ve `source` alanıyla
#      döner, böylece okuyan hangi büyüklüğü aldığını bilir.
#   3. Ağ çağrısı PAHALIDIR ve `.info` kırılgandır → disk önbelleği (varsayılan 7 gün).
#      Float günlerce değişmez; günde bir kez sormak bile fazladır.
_FLOAT_CACHE = DATA_DIR / "float_shares.json"
_FLOAT_TTL_SEC = 7 * 24 * 3600


def _float_cache_read() -> dict:
    try:
        return json.loads(_FLOAT_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def float_shares(symbol: str, *, allow_outstanding: bool = False,
                 fetch: bool = False,
                 max_age_sec: float = _FLOAT_TTL_SEC) -> dict | None:
    """Halka açık hisse adedi: `{"shares", "source", "asof"}` — ölçülemezse None.

    `source` ∈ {"float", "outstanding"} (ikincisi yalnız `allow_outstanding=True` ile).
    Önbellek `data/float_shares.json`; ağ hatası/eksik alan sessizce None'a düşer ve
    ÇAĞIRAN AKIŞI BOZULMAZ (Madde 45 fail-soft deseni).

    **`fetch` VARSAYILAN FALSE ve bu bir mimari karardır (Tur-35/36).** Bu fonksiyonun
    tüketicisi bir PANO UCUdur; `yfinance.Ticker.get_info()` saniyeler sürebilen bir ağ
    çağrısıdır ve onu bir HTTP isteğinin içinde ödemek "ölçüm bütçesini OKUYAN da öder"
    kuralının ihlalidir. Okuyucular önbelleği okur; önbelleği DOLDURMAK ayrı ve BİLİNÇLİ
    bir adımdır:  `python -m nihai.data.market_structure --warm AAPL,MSFT`."""
    sym = str(symbol).strip().upper()
    if not sym:
        return None
    now = time.time()
    cache = _float_cache_read()
    hit = cache.get(sym)
    if isinstance(hit, dict) and (now - float(hit.get("ts", 0))) < max_age_sec:
        if hit.get("shares") is None:
            return None                      # NEGATİF sonuç da önbelleklenir (tekrar sorma)
        if hit.get("source") == "outstanding" and not allow_outstanding:
            return None
        return {"shares": float(hit["shares"]), "source": hit.get("source", "float"),
                "asof": hit.get("asof", "")}

    if not fetch:
        return None          # önbellekte yok ve çekmek İSTENMEDİ → ölçüm yok
    shares: float | None = None
    source = "float"
    try:
        import yfinance as yf
        info = yf.Ticker(sym).get_info() or {}
        for key, src in (("floatShares", "float"), ("sharesOutstanding", "outstanding")):
            v = info.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv > 0:
                shares, source = fv, src
                break
    except Exception:                                            # noqa: BLE001
        shares = None

    cache[sym] = {"shares": shares, "source": source, "ts": now,
                  "asof": time.strftime("%Y-%m-%d", time.localtime(now))}
    try:
        from nihai.atomic_io import atomic_write_text
        _FLOAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(_FLOAT_CACHE, json.dumps(cache, ensure_ascii=False))
    except Exception:                                            # noqa: BLE001
        pass
    if shares is None or (source == "outstanding" and not allow_outstanding):
        return None
    return {"shares": shares, "source": source, "asof": cache[sym]["asof"]}


# ---------------------------------------------------------------------------
# Madde 24 — Halt / LULD sezgisel tespiti
# ---------------------------------------------------------------------------
def warm_float_cache(symbols: list[str]) -> dict:
    """Float önbelleğini DOLDUR (ağ çağrısı; bilinçli adım). Karne döndürür."""
    ok = miss = 0
    for s in symbols:
        r = float_shares(s, fetch=True, allow_outstanding=True)
        if r is None:
            miss += 1
        else:
            ok += 1
            print(f"  {s:<8} {r['shares']:>15,.0f}  ({r['source']})")
    return {"ok": ok, "miss": miss}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Halka açık hisse adedi (float) önbelleği")
    ap.add_argument("--warm", default="", help="virgülle ayrık semboller (ağ çağrısı)")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.warm.split(",") if s.strip()]
    if not syms:
        cache = _float_cache_read()
        dolu = sum(1 for v in cache.values() if isinstance(v, dict) and v.get("shares"))
        print(f"[float] önbellek: {len(cache)} sembol, {dolu} ölçülü ({_FLOAT_CACHE})")
        return
    res = warm_float_cache(syms)
    print(f"[float] ok={res['ok']} ölçülemeyen={res['miss']}")


def halt_or_luld_guard(df: pd.DataFrame) -> tuple[bool, str]:
    """Olası halt veya LULD/anormal bar tespiti.

    (True, reason) → guard aktif, yeni giriş yapma.

    Sezgiseller (gecikmeli veriyle yaklaşık):
      - Donma: son N bar fiyatı sabit + hacim sıfır → olası halt.
      - LULD: son barda |getiri| eşik üstü → bant/anormal hareket.
    """
    ms = CONFIG.market_structure
    if df is None or df.empty or len(df) < max(ms.halt_frozen_bars, 2):
        return False, "CLEAR"

    closes = df["close"].astype(float)
    vols = df["volume"].astype(float) if "volume" in df.columns else None

    # Donma / olası halt
    tail = closes.iloc[-ms.halt_frozen_bars:]
    frozen_price = bool(tail.nunique() == 1)
    zero_vol = bool(vols is not None and (vols.iloc[-ms.halt_frozen_bars:] <= 0).all())
    if frozen_price and zero_vol:
        return True, f"POSSIBLE_HALT_{ms.halt_frozen_bars}_BAR_FROZEN"

    # LULD / anormal tek-bar hareketi
    prev = float(closes.iloc[-2])
    last = float(closes.iloc[-1])
    if prev > 0:
        move = abs(last - prev) / prev
        if move >= ms.luld_move_pct:
            return True, f"LULD_ANOMALY_MOVE_{move*100:.1f}%"

    return False, "CLEAR"


# ---------------------------------------------------------------------------
# Madde 30 — SSR (Short Sale Restriction / Reg SHO Rule 201)
# ---------------------------------------------------------------------------
def ssr_active(prev_close: float, current_price: float) -> bool:
    """Dayanak önceki kapanışa göre eşik kadar (varsayılan %10) düştüyse SSR tetiklenir."""
    if prev_close <= 0:
        return False
    drop = (prev_close - current_price) / prev_close
    return drop >= CONFIG.regulation.ssr_trigger_drop_pct


def ssr_short_guard(prev_close: float, current_price: float) -> tuple[bool, str]:
    """SHORT sinyalleri için SSR farkındalığı.

    (True, reason) → SSR aktif. PDF Madde 30: SSR tetiklendiğinde açığa satış
    yalnızca NBB'nin ÜSTÜNDEN yapılabilir. Sinyal-only modda NBB uptick'i
    modelleyemediğimiz için short sinyali bloke ETMEZ, ama "kısıtlı" olarak
    işaretler ki canlı uygulamada elle/emir katmanında uptick kuralı uygulansın.
    """
    if ssr_active(prev_close, current_price):
        drop = (prev_close - current_price) / prev_close * 100
        return True, f"SSR_ACTIVE_DROP_{drop:.1f}%_SHORT_UPTICK_ONLY"
    return False, "CLEAR"


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    main()
