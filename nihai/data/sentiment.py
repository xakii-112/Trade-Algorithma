"""
sentiment.py — Reddit retail-sentiment ADAY sinyali (C2). PRAW (Reddit API) + VADER.

DÜRÜSTLÜK / DİSİPLİN (docs/EDGE_FINDINGS.md):
- Bu bir KENAR (alpha) İDDİASI DEĞİLDİR. ADAY sinyaldir: önce ölç (research/sentiment_test.py),
  iki periyot + al-tut beta + survivorship. PEAD §16 dersi (survivorship deflasyonu) burada da
  geçerli — ama daha derin bir SINIR var (aşağıda). ÖLÇ, SHIP ETME.
- Fiyat-göstergesi DEĞİL; davranışsal/olay-bazlı (perakende konuşma yoğunluğu+tonu).
- Yön belirsiz: aşırı boğa retail sentimenti hem MOMENTUM hem KONTRARYEN (fade) olabilir;
  varsayılan yorum momentum (LONG), ama test gerçeği gösterir. allow_contrarian ile ters çevrilir.

KRİTİK SINIR — neden gerçek tarihsel backtest YOK:
- Reddit'in ÜCRETSİZ tarihsel API'si yoktur (Pushshift kapandı). PRAW yalnız GÜNCEL/son
  gönderileri verir. Dolayısıyla bu modül iv_history.py gibi İLERİYE-BİRİKİMLİ çalışır:
  her gün anlık snapshot'ı data/sentiment_history/<SYM>.csv'ye ekler; zamanla seri oluşur.
- Yani sentiment_test.py SOĞUK başlangıçta işlem üretemez — iv_history'nin "confidence"
  kademeleri gibi "yeterli geçmiş yok" raporlar. Gerçek ölçüm ancak D2 (7/24 deploy) +
  periyodik `update_sentiment()` ile geçmiş biriktikten sonra mümkün. Bu, dürüst tasarımın
  bedeli — ücretli tarihsel sosyal veri satın alınmıyor (veri bütçesi kararı).

Reddit app credential (ücretsiz): https://www.reddit.com/prefs/apps → 'script' tipi app →
.env'e: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR

SENT_DIR = DATA_DIR / "sentiment_history"
SENT_DIR.mkdir(parents=True, exist_ok=True)

# Ücretsiz Reddit 'script' app credential (.env). Yoksa modül sessizce devre-dışı.
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "NihaiAlgoritma sentiment script")
# Taranan finans subredditleri (virgülle özelleştir: SENTIMENT_SUBREDDITS=...)
SUBREDDITS = [s.strip() for s in os.getenv(
    "SENTIMENT_SUBREDDITS", "wallstreetbets,stocks,investing,StockMarket").split(",") if s.strip()]

_reddit_client = None          # PRAW Reddit (lazy, cache)
_analyzer = None               # VADER (lazy, cache)
_history_cache: dict[str, pd.DataFrame | None] = {}


@dataclass
class SentimentSnapshot:
    """Anlık Reddit sentiment ölçümü. score ∈ [-1, +1] (VADER compound ağırlıklı ort.)."""
    symbol: str
    score: float = 0.0
    n_posts: int = 0
    enabled: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "score": self.score, "n_posts": self.n_posts,
                "enabled": self.enabled, "reason": self.reason}


def _path(symbol: str):
    return SENT_DIR / f"{symbol.upper()}.csv"


def _get_reddit():
    """Lazy PRAW client. Credential yok / praw yok → None (sessiz devre-dışı)."""
    global _reddit_client
    if _reddit_client is not None:
        return _reddit_client
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        logger.debug("[sentiment] REDDIT credential yok → devre-dışı")
        return None
    try:
        import praw
        _reddit_client = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False,
        )
        _reddit_client.read_only = True
        return _reddit_client
    except Exception as e:
        logger.debug(f"[sentiment] praw init başarısız: {e}")
        return None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def fetch_sentiment(symbol: str, *, limit: int = 60,
                    subreddits: list[str] | None = None) -> SentimentSnapshot:
    """CANLI: sembolü anan son gönderileri çek, VADER compound ile skorla, upvote-ağırlıklı
    ortalama döndür. Ağ ister. Credential/praw yoksa enabled=False, score=0 (nötr)."""
    reddit = _get_reddit()
    if reddit is None:
        return SentimentSnapshot(symbol, reason="REDDIT_DISABLED")
    subs = subreddits or SUBREDDITS
    sym = symbol.upper()
    analyzer = _get_analyzer()
    # $TICKER ve TICKER eşleşmesini ara — başka kelimelerle karışmayı azalt.
    query = f'"{sym}" OR "${sym}"'
    scores: list[float] = []
    weights: list[float] = []
    try:
        sub = reddit.subreddit("+".join(subs))
        for post in sub.search(query, sort="new", time_filter="week", limit=limit):
            text = f"{post.title or ''} {getattr(post, 'selftext', '') or ''}"
            if sym not in text.upper():
                continue
            comp = analyzer.polarity_scores(text[:2000])["compound"]
            w = max(1.0, float(getattr(post, "score", 1) or 1))   # upvote ağırlığı (≥1)
            scores.append(comp)
            weights.append(w)
    except Exception as e:
        logger.debug(f"[sentiment] {sym} arama başarısız: {e}")
        return SentimentSnapshot(sym, reason="FETCH_ERROR")
    if not scores:
        return SentimentSnapshot(sym, n_posts=0, enabled=True, reason="NO_POSTS")
    wsum = sum(weights)
    # Tur-81: `scores`/`weights` aynı döngüde birlikte doldurulur → uzunlukları eştir;
    # `strict=True` bu sözleşmeyi MAKİNEYE çevirir (sessiz kırpma yerine hata).
    score = sum(s * w for s, w in zip(scores, weights, strict=True)) / wsum if wsum else 0.0
    return SentimentSnapshot(sym, score=round(score, 4), n_posts=len(scores),
                             enabled=True, reason="OK")


def append_today(symbol: str, snap: SentimentSnapshot) -> None:
    """Bugünün snapshot'ını geçmişe ekle (idempotent — aynı gün üst üste yazılır).
    iv_history deseni: ileriye-birikimli CSV (date,score,n_posts)."""
    if not snap.enabled or snap.n_posts <= 0:
        return
    p = _path(symbol)
    today_iso = date.today().isoformat()
    row = {"date": today_iso, "score": snap.score, "n_posts": snap.n_posts}
    if p.exists():
        df = pd.read_csv(p)
        if (df["date"] == today_iso).any():
            for k, v in row.items():
                df.loc[df["date"] == today_iso, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.sort_values("date", inplace=True)
    df.to_csv(p, index=False)
    _history_cache.pop(symbol.upper(), None)


def update_sentiment(symbol: str, **kw) -> SentimentSnapshot:
    """fetch + append — canlı döngü/cron için tek çağrı (geçmişi büyütür)."""
    snap = fetch_sentiment(symbol, **kw)
    append_today(symbol, snap)
    return snap


def load_history(symbol: str) -> pd.DataFrame | None:
    """Birikmiş sentiment geçmişi (date-index'li). Yoksa None. Cache'li."""
    s = symbol.upper()
    if s in _history_cache:
        return _history_cache[s]
    p = _path(s)
    if not p.exists():
        _history_cache[s] = None
        return None
    try:
        df = pd.read_csv(p)
        df.index = [pd.Timestamp(d).date() for d in df["date"]]
        df = df.sort_index()
        _history_cache[s] = df
        return df
    except Exception:
        _history_cache[s] = None
        return None


def sentiment_signal(history: pd.DataFrame | None, prev_date,
                     *, score_min: float = 0.5, min_posts: int = 5,
                     allow_contrarian: bool = False) -> tuple[str, list[str]]:
    """SAF karar: önceki bar (prev_date) bir sentiment ZİRVESİ günü müydü?

    Dönüş: ('LONG'|'SHORT'|'WAIT', reasons). Varsayılan: yüksek-boğa + yeterli hacim → LONG
    (momentum yorumu). allow_contrarian=True → boğa zirvesini fade (SHORT) / ayı zirvesini LONG.
    min_posts gürültü filtresi (az gönderi = güvenilmez)."""
    if history is None or history.empty:
        return "WAIT", ["NO_SENTIMENT_DATA"]
    if prev_date not in history.index:
        return "WAIT", ["NO_SENTIMENT_EVENT"]
    row = history.loc[prev_date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    score = float(row.get("score", 0.0))
    n = int(row.get("n_posts", 0))
    if n < min_posts:
        return "WAIT", [f"SENTIMENT_LOW_VOLUME_{n}"]
    bullish = score >= score_min
    bearish = score <= -score_min
    if not allow_contrarian:
        if bullish:
            return "LONG", [f"SENTIMENT_BULL_{score:+.2f}", f"posts_{n}"]
        if bearish:
            return "SHORT", [f"SENTIMENT_BEAR_{score:+.2f}", f"posts_{n}"]
    else:
        if bullish:
            return "SHORT", [f"SENTIMENT_FADE_BULL_{score:+.2f}", f"posts_{n}"]
        if bearish:
            return "LONG", [f"SENTIMENT_FADE_BEAR_{score:+.2f}", f"posts_{n}"]
    return "WAIT", [f"SENTIMENT_NEUTRAL_{score:+.2f}"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Reddit sentiment (PRAW+VADER) anlık ölçüm")
    ap.add_argument("symbol")
    ap.add_argument("--save", action="store_true", help="geçmişe ekle (update_sentiment)")
    args = ap.parse_args()
    snap = update_sentiment(args.symbol) if args.save else fetch_sentiment(args.symbol)
    print(snap.as_dict())
    if not snap.enabled:
        print("NOT: REDDIT_CLIENT_ID/SECRET .env'de yok → devre-dışı (reddit.com/prefs/apps).")
