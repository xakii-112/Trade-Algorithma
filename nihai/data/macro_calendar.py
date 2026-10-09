"""
macro_calendar.py — FRED makro 'risk-off' kapısı (B1).

Ücretsiz FRED API'sinden (api.stlouisfed.org) birkaç makro stres göstergesi çekip basit
bir 'risk-off oy' sayımıyla bir risk ÇARPANI üretir. Bu çarpan risk_manager içinde piyasa
rejimi (A4) ve adaptif drawdown freni (Madde 9) ile ÇARPIMSAL istiflenir.

DÜRÜSTLÜK / DİSİPLİN (docs/EDGE_FINDINGS.md):
- VARSAYILAN KAPALI (CONFIG.macro.enable_macro_gate=False). Açmadan önce iki bağımsız
  periyotta A/B yap — makro filtre de overfit / whipsaw üretebilir.
- API key yoksa, kapı kapalıysa veya çekim başarısızsa çarpan 1.0 (etkisiz) → sistem
  davranışı HİÇ değişmez. Sessizce dürüst devre-dışı.
- Bu bir KENAR (alpha) iddiası DEĞİLDİR; yalnız bilinen makro stres rejimlerinde riski kısar.

Göstergeler (FRED seri kodları, hepsi ücretsiz):
- VIXCLS       : CBOE VIX kapanış (oynaklık)
- T10Y2Y       : 10y eksi 2y Hazine getiri farkı (negatif = ters eğri / resesyon sinyali)
- BAMLH0A0HYM2 : ICE BofA US High-Yield OAS kredi spreadi (% — kredi stresi ↑)

FRED key (ücretsiz): https://fred.stlouisfed.org/docs/api/api_key.html → .env'de FRED_API_KEY=
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from loguru import logger

from nihai.config import CONFIG

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Risk-off oyu için izlenen seriler
SERIES = {
    "vix": "VIXCLS",
    "t10y2y": "T10Y2Y",
    "hy_spread": "BAMLH0A0HYM2",
}


@dataclass
class MacroSnapshot:
    """Anlık makro değerlendirme. multiplier=1.0 → etkisiz (risk-on / devre-dışı)."""
    multiplier: float = 1.0
    label: str = "—"
    reasons: list[str] = field(default_factory=list)
    values: dict[str, float | None] = field(default_factory=dict)
    enabled: bool = False
    as_of: float = 0.0   # time.time() epoch

    def as_dict(self) -> dict:
        return {
            "multiplier": self.multiplier, "label": self.label, "reasons": self.reasons,
            "values": self.values, "enabled": self.enabled, "as_of": self.as_of,
        }


def _fetch_latest(series_id: str, api_key: str, *, timeout: float = 8.0) -> float | None:
    """FRED'den serinin en son geçerli gözlemini çeker. Hata/eksikte None (sessiz)."""
    try:
        import requests
    except Exception:
        logger.debug("[macro] requests yok → FRED atlandı")
        return None
    try:
        params: dict[str, str | int] = {
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "sort_order": "desc", "limit": 5,
        }
        r = requests.get(FRED_BASE, params=params, timeout=timeout)
        r.raise_for_status()
        for obs in r.json().get("observations", []):
            val = obs.get("value")
            if val not in (None, "", "."):          # FRED eksik gözlemi "." ile işaretler
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None
    except Exception as e:
        logger.debug(f"[macro] {series_id} çekilemedi: {e}")
        return None


def compute_macro_multiplier(
    vix: float | None, t10y2y: float | None, hy_spread: float | None, cfg=None
) -> tuple[float, str, list[str]]:
    """SAF fonksiyon: gösterge değerlerinden (mult, label, reasons). None'lar atlanır.
    0 tetik → 1.0/RISK_ON, 1 tetik → one_trigger_mult/CAUTION, ≥2 → multi_trigger_mult/RISK_OFF."""
    cfg = cfg or CONFIG.macro
    reasons: list[str] = []
    if vix is not None and vix >= cfg.vix_high:
        reasons.append(f"VIX {vix:.0f}≥{cfg.vix_high:.0f}")
    if t10y2y is not None and t10y2y < cfg.yield_inverted_below:
        reasons.append(f"Ters eğri T10Y2Y {t10y2y:+.2f}")
    if hy_spread is not None and hy_spread >= cfg.hy_spread_high:
        reasons.append(f"HY spread {hy_spread:.1f}≥{cfg.hy_spread_high:.1f}")
    n = len(reasons)
    if n == 0:
        return 1.0, "RISK_ON", reasons
    if n == 1:
        return cfg.one_trigger_mult, "CAUTION", reasons
    return cfg.multi_trigger_mult, "RISK_OFF", reasons


# Modül-içi önbellek (refresh_minutes'te bir gerçek çekim yapılır)
_cache: MacroSnapshot | None = None


def _neutral(reason: str) -> MacroSnapshot:
    return MacroSnapshot(multiplier=1.0, label="—", enabled=False, reasons=[reason])


def get_macro_snapshot(*, force: bool = False) -> MacroSnapshot:
    """Geçerli makro snapshot'ı (önbellekli). Kapı kapalı / key yok → nötr (1.0, devre-dışı),
    AĞ İSTEĞİ YAPILMAZ. Açık + key var → refresh_minutes'te bir FRED'den çeker."""
    global _cache
    cfg = CONFIG.macro

    if not cfg.enable_macro_gate:
        return _neutral("MACRO_GATE kapalı")
    if not cfg.fred_api_key:
        return _neutral("FRED_API_KEY yok")

    now = time.time()
    if (not force and _cache is not None
            and (now - _cache.as_of) < cfg.refresh_minutes * 60):
        return _cache

    vals = {k: _fetch_latest(sid, cfg.fred_api_key) for k, sid in SERIES.items()}
    if all(v is None for v in vals.values()):
        # Çekim tümüyle başarısız → nötr kal (önceki cache varsa onu koru)
        snap = _cache or MacroSnapshot(multiplier=1.0, label="—", enabled=True,
                                       reasons=["FRED çekilemedi → nötr"])
        snap.values = vals
        return snap

    mult, label, reasons = compute_macro_multiplier(
        vals["vix"], vals["t10y2y"], vals["hy_spread"], cfg)
    _cache = MacroSnapshot(multiplier=mult, label=label, reasons=reasons,
                           values=vals, enabled=True, as_of=now)
    logger.info(f"[macro] {label} ×{mult} — {', '.join(reasons) or 'tetik yok'}")
    return _cache
