"""api_orderflow.py — Emir Akışı sekmesinin tek ucu (Tur-103). KARAR-DIŞI, fail-soft.

`GET /api/orderflow?symbol=AAPL&minutes=60&bar=60&feed=sip&step=0&bubbles=150&min_size=0`
→ footprint barları + pencere hacim profili + büyük agresif emir baloncukları (+ kripto sembolünde
likidite ısı haritası). Hesap ve kaynaklar `nihai/data/orderflow.py` docstring'inde. Hata 500 atmaz:
`{"ok": false, "reason": ...}` döner ve pano nedeni basar.

`GET /api/orderflow/gex?symbol=SPY` (Tur-104) → çok vadeli net gamma pozlaması: strike başına call/put
GEX, flip, call/put duvarı. Hisse: panonun zincir sağlayıcısı (yfinance tabanı), ≤45 gün, en çok 8 vade,
10 dk önbellek. Kripto: Bybit + OKX, tüm vadeler, 2 dk önbellek.
"""
from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter
from loguru import logger

from nihai.data import orderflow as OF

router = APIRouter()
_SEMBOL = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,19}")
EN_COK_BAR = 480


@router.get("/api/orderflow")
def orderflow(symbol: str = "AAPL", minutes: int = 60, bar: int = 60, feed: str = "sip",
              step: float = 0.0, bubbles: int = 150, min_size: float = 0.0) -> dict[str, Any]:
    # `def` (async DEĞİL): gövde ağ bekler → FastAPI thread havuzunda koşar, olay döngüsünü dondurmaz (Tur-66).
    sym = (symbol or "").strip().upper()
    if not _SEMBOL.fullmatch(sym):
        return {"ok": False, "reason": "Geçersiz sembol"}
    if bar not in OF.BAR_SN:
        return {"ok": False, "reason": f"bar saniyesi {OF.BAR_SN} içinden olmalı"}
    feed = (feed or "").lower()
    if feed not in ("sip", "iex"):
        return {"ok": False, "reason": "feed: sip | iex"}
    if not (math.isfinite(step) and step >= 0):
        return {"ok": False, "reason": "Adım ≥ 0 olmalı (0 = otomatik)"}
    if not (math.isfinite(min_size) and min_size >= 0):
        return {"ok": False, "reason": "En küçük işlem boyu ≥ 0 olmalı"}
    minutes = max(5, min(OF.EN_COK_DAKIKA, int(minutes)))
    if minutes * 60 // bar > EN_COK_BAR:
        return {"ok": False, "reason": f"{minutes} dk / {bar} sn = {minutes * 60 // bar} bar; en çok {EN_COK_BAR} "
                                       "— daha büyük bar seçin"}
    try:
        return OF.emir_akisi(sym, dakika=minutes, bar_sn=bar, feed=feed, adim=step,
                             baloncuk=max(0, min(500, int(bubbles))), en_az=min_size)
    except Exception as e:
        logger.warning(f"[orderflow] {sym}: {type(e).__name__}: {e}")
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


GEX_VADE_GUN = 45
GEX_EN_COK_VADE = 8
_GEX_ON: dict[str, tuple[float, dict[str, Any]]] = {}


def _hisse_gex(sym: str) -> dict[str, Any]:
    from nihai.web.api_quant import _expiry_years, _provider

    prov = _provider()
    ilk = prov.get_option_chain(sym, None) or {}
    spot = float(ilk.get("underlying_price") or 0.0)
    if not spot > 0:
        return {"ok": False, "reason": f"{sym}: opsiyon zinciri / dayanak fiyatı yok"}
    vadeler = []
    for e in sorted(str(x) for x in (ilk.get("all_expiries") or [])):
        t = _expiry_years(e) or 0.0
        if 0 < t <= GEX_VADE_GUN / 365:
            vadeler.append((e, t))
    vadeler = vadeler[:GEX_EN_COK_VADE]
    if not vadeler:
        return {"ok": False, "reason": f"{sym}: {GEX_VADE_GUN} gün içinde vade yok"}
    with ThreadPoolExecutor(4) as ex:
        zincir = list(ex.map(lambda v: prov.get_option_chain(sym, v[0]) or {}, vadeler))
    g = OF.gex_toplu([(e, t, z.get("calls"), z.get("puts")) for (e, t), z in zip(vadeler, zincir, strict=True)],
                     spot, r=0.04, kontrat=100)
    if g is None:
        return {"ok": False, "reason": f"{sym}: hiçbir vadede IV + OI iki tarafta birlikte yok"}
    kaynak = sorted({str(z.get("option_source") or "yfinance") for z in zincir if z})
    g.update({"ok": True, "piyasa": "hisse", "kaynak": " + ".join(kaynak) + " zinciri (OI günlük — önceki seans sonu)",
              "uyari": [], "vade_ufku": f"≤{GEX_VADE_GUN} gün, en çok {GEX_EN_COK_VADE} vade", "yenile_sn": 600})
    return g


@router.get("/api/orderflow/gex")
def orderflow_gex(symbol: str = "SPY") -> dict[str, Any]:
    # `def`: zincir istekleri ağ bekler (Tur-66). Önbellek yalnız başarılı sonucu tutar.
    sym = (symbol or "").strip().upper()
    if not _SEMBOL.fullmatch(sym):
        return {"ok": False, "reason": "Geçersiz sembol"}
    coin = OF.kripto_tabani(sym)
    hit = _GEX_ON.get(sym)
    if hit and time.monotonic() - hit[0] < (120 if coin else 600):
        return hit[1]
    try:
        out = OF.kripto_gex(coin) if coin else _hisse_gex(sym)
    except Exception as e:
        logger.warning(f"[orderflow/gex] {sym}: {type(e).__name__}: {e}")
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    out["symbol"] = sym
    out["zaman"] = int(time.time() * 1000)
    if out.get("ok"):
        _GEX_ON[sym] = (time.monotonic(), out)
    return out
