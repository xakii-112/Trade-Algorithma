"""
episode_resolver.py — GÖLGE kararların SONUÇ kolunu kapatan çözücü (2026-07-26, B-3).

Neden var: `decision._record_shadow` her gölge karar için `episodic.append_episode(...,
outcome=None)` yazıyordu ve `episodic.resolve_episode(id, outcome)` API'si HAZIRDI —
ama onu ÇAĞIRAN taraf yoktu. 2026-07-26 denetiminde 248 epizodun 248'inde `outcome=None`
ölçüldü: epizodik hafıza bir ARŞİV'di, öğrenme organı değil. Sonuç yazılmadığı için
`similar_episodes` sonuçsuz komşular döndürüyor (`_outcome_vector` → nötr (0,0)),
`consolidate` bilgi taşımayan kümeler üretiyor ve `_forget_score` her epizodu
"çözülmemiş" (0.3) değerinde tutuyordu.

Ne yapar: `kind="ai_decision"` epizotlarından `outcome`'u boş VE yeterince eskimiş
olanları alır, sembolün GÜNLÜK barlarını çeker, karar barından `horizon` bar sonrasına
kadarki getiriyi hesaplar ve `episodic.resolve_episode` ile bağlar.

LOOK-AHEAD YASAĞI (Madde 3) BOZULMAZ: burada geleceğe bakmak SERBESTTİR ve zorunludur —
bu bir SONUÇ ÖLÇÜMÜdür, özellik üretimi değil. Koruma şudur: karar barı epizodun ts'inden
ÖNCEKİ (veya ts'e eşit) son bar olarak seçilir; ileri getiri yalnız o bardan SONRAKİ
barlardan hesaplanır ve `horizon` bar dolmadıysa epizot ÇÖZÜLMEZ (bekler).

DÜRÜSTLÜK: ABSTAIN epizotlarında taraf yoktur → `win=None` (nötr) yazılır, ileri getiri
yine kaydedilir ("kaçırılan/kaçınılan hareket ne kadardı" ölçülebilsin). ABSTAIN'i
uydurma bir "doğru karar" etiketiyle kazanç saymak ACI DERS #3 disiplinine aykırıdır.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from nihai.ai import episodic
from nihai.console import ensure_utf8_stdout

# Varsayılan ufuk: 5 GÜNLÜK bar (~1 işlem haftası). Gölge kararlar günlük barlarda
# üretildiğinden (shadow_runner `_fetch_bars(..., "1d")`) ölçüm de günlük bardadır.
DEFAULT_HORIZON_BARS = int(os.getenv("AI_EPISODE_HORIZON_BARS", "5"))
# Barları çekmek için gereken geçmiş: ufuk + karar barını bulmaya yetecek pencere.
_LOOKBACK_BARS = 260
# Sonucun "kazanç" sayılması için asgari mutlak getiri (gürültü bandı). 0 = kapalı.
MIN_ABS_RET = float(os.getenv("AI_EPISODE_MIN_ABS_RET", "0.0"))

_SIDE_FROM_TAG = {"long_bias": 1, "short_bias": -1}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _side_of(ep: episodic.Episode) -> int:
    """Epizodun tarafı: +1 LONG_BIAS, -1 SHORT_BIAS, 0 ABSTAIN/bilinmiyor.
    `decision._record_shadow` tags=[strategy, action.lower()] yazar."""
    for t in ep.tags or []:
        s = _SIDE_FROM_TAG.get(str(t).strip().lower())
        if s is not None:
            return s
    return 0


def _bars_index(df: pd.DataFrame) -> pd.DatetimeIndex | None:
    """df'nin tz-aware UTC DatetimeIndex'i (index ya da 'date'/'timestamp' kolonu)."""
    idx: Any = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        col = next((c for c in df.columns if str(c).lower() in ("date", "timestamp", "time")), None)
        if col is None:
            return None
        idx = pd.to_datetime(df[col], errors="coerce", utc=True)
        if not isinstance(idx, pd.DatetimeIndex):
            return None
        return idx
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def _closes(df: pd.DataFrame) -> np.ndarray | None:
    col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if col is None:
        return None
    try:
        arr = df[col].to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


def _forward_return(df: pd.DataFrame, ts: datetime, horizon: int) -> tuple[float, str] | None:
    """(ileri getiri, karar-barı ISO) — ufuk dolmadıysa/veri yetmezse None.

    Karar barı: ts'ten SONRA olmayan son bar. İleri fiyat: o barın `horizon` bar
    sonrası. `horizon` bar henüz oluşmadıysa None döner (epizot bekler)."""
    idx = _bars_index(df)
    closes = _closes(df)
    if idx is None or closes is None or len(idx) != len(closes):
        return None
    mask = idx <= pd.Timestamp(ts)
    if not bool(mask.any()):
        return None
    i = int(np.flatnonzero(np.asarray(mask))[-1])
    j = i + horizon
    if j >= len(closes):
        return None            # ufuk henüz dolmadı — bir sonraki koşuda çözülür
    p0, p1 = float(closes[i]), float(closes[j])
    if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
        return None
    return (p1 / p0 - 1.0), str(idx[i].isoformat())


def resolve_pending(horizon_bars: int = DEFAULT_HORIZON_BARS, days: int = 90,
                    max_episodes: int = 300, provider: Any = None) -> dict:
    """Çözülmemiş `ai_decision` epizotlarını ileri getiriyle bağla.

    Dönen: {'aday','cozulen','bekleyen','veri_yok','kazanan','kaybeden','notr',
            'ort_ileri_getiri','horizon_bars'} — hiçbir kenar iddiası taşımaz."""
    out: dict[str, Any] = {"aday": 0, "cozulen": 0, "bekleyen": 0, "veri_yok": 0,
                           "kazanan": 0, "kaybeden": 0, "notr": 0,
                           "ort_ileri_getiri": None, "horizon_bars": int(horizon_bars)}
    try:
        eps = episodic.recent_episodes(kind="ai_decision", days=days, limit=5000)
    except Exception as e:  # noqa: BLE001 — çözücü hiçbir akışı düşürmemeli
        logger.warning(f"[ai.episode_resolver] epizotlar okunamadı: {e}")
        return out

    pending = [e for e in eps if e.outcome is None and e.symbol][:max_episodes]
    out["aday"] = len(pending)
    if not pending:
        return out

    if provider is None:
        try:
            from nihai.data.data_provider import get_provider
            provider = get_provider()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ai.episode_resolver] sağlayıcı alınamadı: {e}")
            return out

    bars_cache: dict[str, pd.DataFrame | None] = {}
    rets: list[float] = []
    for ep in pending:
        sym = str(ep.symbol)
        if sym not in bars_cache:
            try:
                bars_cache[sym] = provider.get_bars(sym, "1d", lookback_bars=_LOOKBACK_BARS)
            except Exception as e:  # noqa: BLE001 — tek sembol hatası koşuyu düşürmemeli
                logger.debug(f"[ai.episode_resolver] {sym} bar çekimi hata: {e}")
                bars_cache[sym] = None
        df = bars_cache[sym]
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            out["veri_yok"] += 1
            continue
        ts = episodic._parse_ts(ep.ts)
        if ts is None:
            out["veri_yok"] += 1
            continue
        res = _forward_return(df, ts, int(horizon_bars))
        if res is None:
            out["bekleyen"] += 1
            continue
        fwd, bar_iso = res
        side = _side_of(ep)
        if side == 0:
            win: bool | None = None                      # ABSTAIN — taraf yok, uydurma etiket YOK
            out["notr"] += 1
        else:
            signed = fwd * side
            if MIN_ABS_RET > 0 and abs(fwd) < MIN_ABS_RET:
                win = None                               # gürültü bandı — sonuç belirsiz
                out["notr"] += 1
            else:
                win = signed > 0
                out["kazanan" if win else "kaybeden"] += 1
        outcome = {"win": win, "fwd_ret_pct": round(fwd * 100.0, 4),
                   "horizon_bars": int(horizon_bars), "side": side,
                   "decision_bar": bar_iso, "resolved_at": _utc_now().isoformat(timespec="seconds")}
        try:
            if episodic.resolve_episode(ep.id, outcome):
                out["cozulen"] += 1
                rets.append(fwd)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ai.episode_resolver] {ep.id} çözülemedi: {e}")

    if rets:
        out["ort_ileri_getiri"] = round(float(np.mean(rets)) * 100.0, 4)
    logger.info(f"[ai.episode_resolver] aday={out['aday']} çözülen={out['cozulen']} "
                f"bekleyen={out['bekleyen']} veri-yok={out['veri_yok']} "
                f"(K/Z/N={out['kazanan']}/{out['kaybeden']}/{out['notr']})")
    return out


def outcome_report(days: int = 90) -> dict:
    """Çözülmüş gölge-karar epizotlarının özeti (KARAR-DIŞI; kenar iddiası YOK)."""
    try:
        eps = episodic.recent_episodes(kind="ai_decision", days=days, limit=5000)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ai.episode_resolver] rapor okunamadı: {e}")
        return {"n": 0}
    # `Episode.outcome` tipi `dict | None` — mypy daraltması için önce dict listesine geç.
    resolved: list[dict] = [e.outcome for e in eps if isinstance(e.outcome, dict)]
    sided: list[dict] = [o for o in resolved if isinstance(o.get("win"), bool)]
    wins = sum(1 for o in sided if bool(o["win"]))
    rets = [float(o["fwd_ret_pct"]) for o in resolved if o.get("fwd_ret_pct") is not None]
    return {
        "n_epizot": len(eps),
        "n_cozulmus": len(resolved),
        "n_tarafli": len(sided),
        "isabet": round(wins / len(sided), 4) if sided else None,
        "ort_ileri_getiri_pct": round(float(np.mean(rets)), 4) if rets else None,
        "cozulme_orani": round(len(resolved) / len(eps), 4) if eps else None,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: `--resolve` (bekleyenleri çöz) · `--report` (özet) · `--horizon N`."""
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(description="Gölge karar epizotlarının sonuç çözücüsü")
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_BARS)
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args(argv)
    if args.resolve:
        print(_json.dumps(resolve_pending(horizon_bars=args.horizon, days=args.days),
                          ensure_ascii=False, indent=2))
    if args.report:
        print(_json.dumps(outcome_report(days=args.days), ensure_ascii=False, indent=2))
    if not (args.resolve or args.report):
        ap.print_help()
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
