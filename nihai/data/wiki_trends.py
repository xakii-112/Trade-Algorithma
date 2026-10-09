"""Wikimedia günlük insan görüntülenmeleri; geçmişe göre ilgi ölçümü, karar dışı."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, date, datetime, timedelta
from urllib.parse import quote

import numpy as np
import pandas as pd
from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR
from nihai.net import contact_ua, shared_client

CACHE_DIR = DATA_DIR / "wiki_cache"
API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


def attention_series(items: list[dict], *, window: int = 28) -> pd.DataFrame | None:
    """Her günün karşılaştırması yalnız ÖNCEKİ window takvim günüdür.

    Eksik gün sıfır sayılmaz; tam geçmiş pencere olmadan oran/z-score yoktur.
    Sabit geçmişte standart sapma sıfırsa z-score None (oran yine ölçülebilir).
    """
    if not isinstance(items, list) or not items or not isinstance(window, int) or not 2 <= window <= 3660:
        return None
    rows = []
    for item in items:
        try:
            day = datetime.strptime(str(item["timestamp"]), "%Y%m%d%H").date()
            views = float(item["views"])
            if np.isfinite(views) and views >= 0 and views.is_integer():
                rows.append((pd.Timestamp(day), views))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if not rows:
        return None
    data = pd.DataFrame(rows, columns=["date", "views"]).drop_duplicates()
    data = data.loc[~data.duplicated("date", keep=False)].set_index("date").sort_index()
    if data.empty:
        return None
    if (data.index.max() - data.index.min()).days > 3660:
        return None
    data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq="D", name="date"))
    prior = data["views"].shift(1).rolling(window, min_periods=window)
    data["baseline_mean"] = prior.mean()
    data["baseline_std"] = prior.std(ddof=1)
    data["ratio"] = data["views"] / data["baseline_mean"].where(data["baseline_mean"] > 0)
    data["zscore"] = (data["views"] - data["baseline_mean"]) / data["baseline_std"].where(data["baseline_std"] > 0)
    return data.replace([np.inf, -np.inf], np.nan)


def report(article: str, *, since: str = "2025-01-01", until: str | None = None,
           project: str = "en.wikipedia.org", fetch: bool = False, force: bool = False) -> dict | None:
    """Gün tamamlandıktan sonra günlük seri; aynı pencere için bir istek, cache-only okuma."""
    try:
        article = article.strip().replace(" ", "_")
        lo = date.fromisoformat(since)
        yesterday = datetime.now(UTC).date() - timedelta(days=1)
        hi = min(date.fromisoformat(until) if until else yesterday, yesterday)
        if (not article or len(article) > 200 or any(ord(c) < 32 for c in article)
                or not re.fullmatch(r"[a-z][a-z0-9-]{0,20}\.wikipedia\.org", project)
                or lo < date(2015, 7, 1) or lo > hi or (hi - lo).days > 3660):
            return None
    except (ValueError, TypeError, AttributeError):
        return None
    key = hashlib.sha256(f"{project}|{article}|{lo}|{hi}".encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    cached = None
    age = float("inf")
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - path.stat().st_mtime
    except (OSError, ValueError):
        pass
    if not fetch or (cached is not None and age < 86400 and not force):
        return {**cached, "cache_state": "fresh" if age < 86400 else "stale"} if isinstance(cached, dict) else None
    try:
        url = f"{API_URL}/{project}/all-access/user/{quote(article, safe='')}/daily/{lo:%Y%m%d}/{hi:%Y%m%d}"
        response = shared_client().get(url, headers={"User-Agent": contact_ua("NihaiWiki/1.0")}, timeout=60)
        response.raise_for_status()
        data = attention_series(response.json().get("items"))
        if data is None:
            raise ValueError("Wikimedia günlük ölçüm yok")
        data = data.loc[(data.index >= pd.Timestamp(lo)) & (data.index <= pd.Timestamp(hi))].copy()
        if data.empty:
            raise ValueError("Wikimedia istenen pencereyi döndürmedi")
        measured = data["views"].dropna()
        data.index = data.index.strftime("%Y-%m-%d")
        rows = data.reset_index().astype(object).where(pd.notna(data.reset_index()), None).to_dict("records")
        result = dict(ok=True, report_type="wiki_trends", article=article, project=project, since=lo.isoformat(), until=hi.isoformat(),
            fetched_at=datetime.now(UTC).isoformat(), source=url, window=28, rows=rows,
            observed_days=len(measured), requested_days=(hi - lo).days + 1,
            note="İnsan görüntülenmeleri; 28 tam geçmiş güne göre oran ve z-score. Eksik günler sıfır sayılmaz; fiyat yönü ölçülmez.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(result, ensure_ascii=False, allow_nan=False))
        return {**result, "cache_state": "downloaded"}
    except Exception as exc:
        logger.warning(f"[wiki] veri yenilenemedi: {type(exc).__name__}")
        return {**cached, "cache_state": "stale", "refresh_failed": True} if isinstance(cached, dict) else None


def _cli() -> None:
    import argparse

    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Wikimedia günlük ilgi geçmişi")
    ap.add_argument("--article", required=True)
    ap.add_argument("--since", default="2025-01-01")
    ap.add_argument("--until", default=None)
    ap.add_argument("--project", default="en.wikipedia.org")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    result = report(args.article, since=args.since, until=args.until, project=args.project,
                    fetch=not args.no_fetch, force=args.force)
    print(json.dumps(result if result is not None else {"ok": False, "reason": "Wikimedia ölçümü yok"}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
