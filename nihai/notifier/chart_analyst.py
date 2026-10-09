"""
chart_analyst.py — AI GRAFİK OKUMA/YORUMLAMA (vision): PNG grafik → yapılandırılmış teknik okuma.

Ne yapar: chart_render'ın ürettiği fiyat grafiğini (PNG) + aynı barlardan hesaplanan
sayısal bağlamı (kapanış, EMA20/50, RSI14, 20-bar aralığı, hacim oranı) bir görüş
(vision) modeline gönderir; model grafiği OKUR ve yapılandırılmış Türkçe teknik okuma
döndürür: trend, momentum, destek/direnç, hacim, formasyon, risk notu, özet.

DÜRÜSTLÜK / SİNYAL-ONLY DİSİPLİNİ (llm_event.py deseniyle aynı):
- Bu bir KENAR (alpha) İDDİASI DEĞİLDİR ve bir SİNYAL KAYNAĞI DEĞİLDİR. Betimleyici
  bir okuma yardımcısıdır (operatör grafiğe bakarken ikinci göz). Çıktı hiçbir karar
  yoluna (strategy/gate) BAĞLANMAZ; yalnız Telegram/CLI'da insana gösterilir.
- Model FİYAT TAHMİNİ yapmaz; yalnız görünen yapıyı betimler. Sayısal bağlam
  görüntü-halüsinasyonunu (yanlış seviye okuma) azaltmak için verilir.

Sağlayıcı: CHART_LLM_PROVIDER (yoksa LLM_EVENT_PROVIDER; 'gemini' ücretsiz katman |
'anthropic'). Görüş gerektirir: gemini-2.5-flash / claude-* çok-modlu.

Kullanım:
    python -m nihai.notifier.chart_analyst AAPL --interval 1h
    Telegram: /yorum AAPL   (grafik + AI okuması birlikte gönderilir)
"""
from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass, field

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import pandas as pd
from loguru import logger

from nihai.config import DATA_PROVIDER
from nihai.data.data_provider import Interval, get_provider

# llm_event ile aynı sağlayıcı mantığı; grafik için ayrı override mümkün.
CHART_LLM_PROVIDER = (os.getenv("CHART_LLM_PROVIDER")
                      or os.getenv("LLM_EVENT_PROVIDER")
                      or "anthropic").lower()
CHART_LLM_MODEL = os.getenv("CHART_LLM_MODEL", "")   # boş → sağlayıcı varsayılanı
_ANTHROPIC_DEFAULT = "claude-haiku-4-5"              # betimleme işi → ucuz model yeter
_GEMINI_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "trend": {"type": "string", "enum": ["yukari", "asagi", "yatay", "belirsiz"]},
        "momentum": {"type": "string", "enum": ["guclu_alici", "alici", "notr", "satici", "guclu_satici"]},
        "destek": {"type": "array", "items": {"type": "number"}, "description": "En fazla 2 destek seviyesi"},
        "direnc": {"type": "array", "items": {"type": "number"}, "description": "En fazla 2 direnç seviyesi"},
        "hacim_yorumu": {"type": "string", "description": "Hacim ne söylüyor (≤140 karakter)"},
        "formasyon": {"type": "string", "description": "Görünen belirgin yapı/formasyon yoksa 'yok' (≤100 karakter)"},
        "risk_notu": {"type": "string", "description": "Okumayı geçersiz kılacak koşul (≤140 karakter)"},
        "ozet": {"type": "string", "description": "2-3 cümle Türkçe genel okuma (≤400 karakter)"},
    },
    "required": ["trend", "momentum", "destek", "direnc", "hacim_yorumu",
                 "formasyon", "risk_notu", "ozet"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a technical chart reader. You are given a price chart image (close line, "
    "EMA20/EMA50 overlays, volume subpanel) and numeric context computed from the SAME bars. "
    "READ the chart and describe ONLY what is visible: trend structure, momentum, the most "
    "relevant support/resistance levels (use the numeric context to anchor price levels — do "
    "not invent numbers outside the visible range), what volume shows, and any obvious pattern. "
    "Do NOT predict future prices, do NOT give buy/sell advice, do NOT claim probabilities. "
    "If the chart is unclear, say so via trend='belirsiz'. Respond in TURKISH field values."
)

_TREND_TR = {"yukari": "Yukarı", "asagi": "Aşağı", "yatay": "Yatay", "belirsiz": "Belirsiz"}
_MOM_TR = {"guclu_alici": "güçlü alıcılı", "alici": "alıcılı", "notr": "nötr",
           "satici": "satıcılı", "guclu_satici": "güçlü satıcılı"}


@dataclass
class ChartRead:
    """Tek grafiğin yapılandırılmış AI okuması. Betimleyicidir — sinyal değildir."""
    trend: str = "belirsiz"
    momentum: str = "notr"
    destek: list[float] = field(default_factory=list)
    direnc: list[float] = field(default_factory=list)
    hacim_yorumu: str = ""
    formasyon: str = "yok"
    risk_notu: str = ""
    ozet: str = ""


def parse_read(data: dict | None) -> ChartRead | None:
    """JSON dict → ChartRead (savunmacı: eksik alan = güvenli varsayılan, sayılar filtreli)."""
    if not isinstance(data, dict):
        return None

    def _nums(v) -> list[float]:
        out = []
        for x in (v if isinstance(v, list) else []):
            try:
                out.append(round(float(x), 2))
            except (TypeError, ValueError):
                continue
        return out[:2]

    trend = str(data.get("trend", "belirsiz"))
    mom = str(data.get("momentum", "notr"))
    return ChartRead(
        trend=trend if trend in _TREND_TR else "belirsiz",
        momentum=mom if mom in _MOM_TR else "notr",
        destek=_nums(data.get("destek")),
        direnc=_nums(data.get("direnc")),
        hacim_yorumu=str(data.get("hacim_yorumu", ""))[:140],
        formasyon=str(data.get("formasyon", "yok"))[:100] or "yok",
        risk_notu=str(data.get("risk_notu", ""))[:140],
        ozet=str(data.get("ozet", ""))[:400],
    )


def numeric_context(df: pd.DataFrame, symbol: str, interval: str) -> str:
    """Aynı barlardan sayısal bağlam metni — modelin seviye halüsinasyonunu azaltır.

    Look-ahead sorunu yok: grafik neyi gösteriyorsa onu özetler (geçmiş barlar)."""
    from nihai.signals.strategy import ema, rsi
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    e20 = float(ema(close, 20).iloc[-1])
    e50 = float(ema(close, 50).iloc[-1])
    try:
        r14 = float(rsi(close, 14).iloc[-1])
    except Exception:
        r14 = float("nan")
    lo20, hi20 = float(df["low"].tail(20).min()), float(df["high"].tail(20).max())
    lo_all, hi_all = float(df["low"].min()), float(df["high"].max())
    vol_note = ""
    if "volume" in df.columns and len(df) >= 20:
        v = df["volume"].astype(float)
        v_avg = float(v.tail(20).mean())
        if v_avg > 0:
            vol_note = f"Son bar hacmi / 20-bar ort. = {float(v.iloc[-1]) / v_avg:.2f}. "
    chg = (last / float(close.iloc[0]) - 1.0) * 100.0 if float(close.iloc[0]) else 0.0
    return (f"{symbol} {interval}, {len(df)} bar. Son kapanış {last:.2f} "
            f"(pencere başından {chg:+.1f}%). EMA20 {e20:.2f}, EMA50 {e50:.2f}. "
            f"RSI14 {r14:.0f}. Son 20 bar aralığı {lo20:.2f}–{hi20:.2f}; "
            f"tüm pencere {lo_all:.2f}–{hi_all:.2f}. {vol_note}"
            f"Destek/direnç değerleri {lo_all:.2f}–{hi_all:.2f} aralığında olmalı.")


def _loads_lenient(text: str) -> dict | None:
    """JSON ayrıştır; ```json``` sarmalı/önek varsa ilk {...} bloğunu ayıkla (llm_event deseni)."""
    if not text:
        return None
    t = text.strip().strip("`")
    if t[:4].lower() == "json":
        t = t[4:]
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Sağlayıcı yolları (llm_event.py desenleri: lazy client, savunmacı, sessiz devre-dışı)
# ---------------------------------------------------------------------------
def _read_anthropic(png: bytes, context: str, model: str) -> ChartRead | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.debug("[chart_analyst] ANTHROPIC_API_KEY yok → devre-dışı")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        logger.debug(f"[chart_analyst] anthropic init başarısız: {e}")
        return None
    try:
        resp = client.messages.create(
            model=model, max_tokens=700, system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": READ_SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(png).decode()}},
                {"type": "text", "text": f"Sayısal bağlam: {context}"},
            ]}],
        )
    except Exception as e:
        logger.warning(f"[chart_analyst] anthropic çağrısı başarısız: {e}")
        return None
    text = next((getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"), "")
    return parse_read(_loads_lenient(text))


_GEMINI_SYSTEM = _SYSTEM + (
    " Respond with ONLY a JSON object (no prose, no markdown fences). Keys EXACTLY: "
    "trend (yukari|asagi|yatay|belirsiz), momentum (guclu_alici|alici|notr|satici|guclu_satici), "
    "destek (array of at most 2 numbers), direnc (array of at most 2 numbers), "
    "hacim_yorumu (string <=140 chars, Turkish), formasyon (string <=100 chars, Turkish, 'yok' if none), "
    "risk_notu (string <=140 chars, Turkish), ozet (string <=400 chars, Turkish)."
)


def _read_gemini(png: bytes, context: str, model: str) -> ChartRead | None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not key:
        logger.debug("[chart_analyst] GEMINI_API_KEY yok → devre-dışı")
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        cfg = types.GenerateContentConfig(
            system_instruction=_GEMINI_SYSTEM,
            response_mime_type="application/json",
            temperature=0.0,
        )
        resp = client.models.generate_content(
            model=model,
            contents=types.Content(role="user", parts=[
                types.Part.from_bytes(data=png, mime_type="image/png"),
                types.Part.from_text(text=f"Sayısal bağlam: {context}"),
            ]),
            config=cfg,
        )
    except Exception as e:
        logger.warning(f"[chart_analyst] gemini çağrısı başarısız: {e}")
        return None
    return parse_read(_loads_lenient(getattr(resp, "text", "") or ""))


def read_chart_png(png: bytes, context: str) -> ChartRead | None:
    """PNG + sayısal bağlam → ChartRead. None = istemci yok / hata."""
    if CHART_LLM_PROVIDER == "gemini":
        return _read_gemini(png, context, CHART_LLM_MODEL or _GEMINI_DEFAULT)
    return _read_anthropic(png, context, CHART_LLM_MODEL or _ANTHROPIC_DEFAULT)


# ---------------------------------------------------------------------------
# Uçtan uca: sembol → (PNG, okuma)  +  Telegram biçimleyici
# ---------------------------------------------------------------------------
def analyze_chart(symbol: str, *, interval: Interval = "1h", lookback: int = 120,
                  provider_name: str | None = None) -> tuple[bytes | None, ChartRead | None]:
    """Sembolün grafiğini üret + AI'a okut. (png, read) döndürür; ikisi de None olabilir."""
    from nihai.notifier.chart_render import render_price_png
    sym = (symbol or "").upper().strip()
    if not sym:
        return None, None
    try:
        provider = get_provider(provider_name or DATA_PROVIDER, with_cache=True)
        df = provider.get_bars(sym, interval, max(40, min(int(lookback), 400)))
    except Exception as e:
        logger.warning(f"[chart_analyst] {sym} veri çekilemedi: {e}")
        return None, None
    if df is None or getattr(df, "empty", True) or len(df) < 20:
        return None, None
    png = render_price_png(sym, interval=interval, lookback=lookback, df=df)
    if png is None:
        return None, None
    read = read_chart_png(png, numeric_context(df, sym, str(interval)))
    return png, read


def format_telegram(read: ChartRead, symbol: str, interval: str) -> str:
    """ChartRead → Telegram başlık metni (Markdown, ≤1000 karakter — caption limiti 1024)."""
    lines = [f"🤖 *AI Grafik Okuması — {symbol} ({interval})*",
             f"Trend: *{_TREND_TR.get(read.trend, read.trend)}* · momentum {_MOM_TR.get(read.momentum, read.momentum)}"]
    if read.destek:
        lines.append("Destek: " + " / ".join(f"`{x:.2f}`" for x in read.destek))
    if read.direnc:
        lines.append("Direnç: " + " / ".join(f"`{x:.2f}`" for x in read.direnc))
    if read.formasyon and read.formasyon.lower() != "yok":
        lines.append(f"Formasyon: {read.formasyon}")
    if read.hacim_yorumu:
        lines.append(f"Hacim: {read.hacim_yorumu}")
    if read.ozet:
        lines.append(read.ozet)
    if read.risk_notu:
        lines.append(f"⚠️ {read.risk_notu}")
    lines.append("_Betimleyici okuma — sinyal/tavsiye değildir._")
    text = "\n".join(lines)
    return text[:1000]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI grafik okuma/yorumlama (vision)")
    ap.add_argument("symbol")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--provider", default=None, help="veri sağlayıcı override (örn. alpaca)")
    ap.add_argument("--save-png", default=None, help="grafiği bu dosyaya da kaydet")
    args = ap.parse_args()

    png, read = analyze_chart(args.symbol, interval=args.interval,
                              lookback=args.lookback, provider_name=args.provider)
    if png and args.save_png:
        with open(args.save_png, "wb") as f:
            f.write(png)
        print(f"PNG → {args.save_png}")
    if read is None:
        print(f"{args.symbol}: AI okuması üretilemedi "
              f"(veri yok / {CHART_LLM_PROVIDER} anahtarı yok / ağ hatası).")
        sys.exit(1)
    print(format_telegram(read, args.symbol.upper(), args.interval).replace("*", "").replace("`", ""))
