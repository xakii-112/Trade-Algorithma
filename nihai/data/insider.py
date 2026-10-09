"""
insider.py — SEC EDGAR Form 4 (insider) açık-piyasa alım sinyali ADAYI (C1).

DÜRÜSTLÜK / DİSİPLİN (docs/EDGE_FINDINGS.md):
- Bu bir KENAR (alpha) İDDİASI DEĞİLDİR. Sadece bir ADAY sinyaldir: önce ölçülür
  (research/insider_test.py → 2 periyot + al-tut beta + pit_universe survivorship),
  PEAD'in §13.5/§16'da yaşadığı survivorship deflasyonu burada da beklenir. ÖLÇ, SHIP ETME.
- Fiyat-göstergesi DEĞİL; temel/olay-bazlı (insider'ların kendi şirketlerini açık piyasada
  satın alması). Bu yüzden fiyat-göstergesi ailesinden bağımsız bir kenar KAYNAĞI olabilir.
- Ücretsiz: SEC EDGAR (data.sec.gov / sec.gov/Archives). Tek gereksinim: kibar bir
  User-Agent (SEC kuralı). Anahtar YOK, ücret YOK.

Look-ahead güvenliği:
- Olaylar FİLİNG (dosyalama) tarihine göre indekslenir, işlem tarihine göre DEĞİL.
  Form 4 işlemden sonra 2 iş günü içinde dosyalanır; bilgi ancak dosyalandığında halka
  açık olur. Giriş = dosyalama tarihinin ERTESİ barı → ileri-bakış yok.

Yöntem:
1. Ticker → CIK  (sec.gov/files/company_tickers.json)
2. CIK → tüm Form 4 dosyalamaları  (data.sec.gov/submissions/CIK##########.json + arşiv parçaları)
3. Her Form 4 XML'i ayrıştır → açık-piyasa ALIM (kod 'P') vs SATIM (kod 'S') hisse/değer.
   (Grant/exercise/gift/tax kodları = gürültü, atlanır.) Dosyalama tarihine göre topla.
4. Sembol başına olay tablosu data/insider_cache/<SYM>.json'a yazılır (EDGAR'ı tekrar
   yormamak için; backtest sepet×yıl üzerinde binlerce istek olurdu).

SINIR: ücretsiz veri PIT değil — şirket isim/CIK değişiklikleri, delist olmuş tickerların
eksik submissions'ı vb. survivorship'i tam çözmez. Gerçek altın standart yine Sharadar/CRSP.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import pandas as pd
from loguru import logger

from nihai import net
from nihai.config import DATA_DIR

INSIDER_DIR = DATA_DIR / "insider_cache"
INSIDER_DIR.mkdir(parents=True, exist_ok=True)

# SEC, her isteğin kendini tanıtan bir User-Agent taşımasını ŞART koşar (e-posta dahil).
# Dize `nihai.net.contact_ua` ile ÜRETİLİR (.env'deki SEC_USER_AGENT) — kimlik kodda DEĞİL.
_THROTTLE = float(os.getenv("SEC_THROTTLE", "0.2"))   # ~5 req/s (SEC limiti 10/s)
_request_lock = threading.Lock()
_retry_after = 0.0

CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_cik_map: dict[str, int] | None = None
_events_cache: dict[str, pd.DataFrame | None] = {}


def _headers() -> dict[str, str]:
    return {"User-Agent": net.contact_ua("NihaiAlgoritma research"),
            "Accept-Encoding": "gzip, deflate"}


def _get(url: str, *, as_json: bool = True, timeout: float = 15.0):
    """EDGAR GET (kibar throttle + tek deneme). Hata → None (sessiz, çağıran karar verir)."""
    global _retry_after
    try:
        import requests
    except Exception:
        logger.debug("[insider] requests yok → EDGAR atlandı")
        return None
    try:
        # ponytail: süreç içi seri ağ erişimi; çok süreçli toplu indirme gerekirse
        # tek EDGAR indirme servisine taşı. Tüm çağıranlar AYNI kuyruğu paylaşır.
        with _request_lock:
            time.sleep(max(0.11, _THROTTLE, _retry_after - time.monotonic()))
            r = requests.get(url, headers=_headers(), timeout=timeout)
            if r.status_code == 429:
                raw = r.headers.get("Retry-After", "30")
                try:
                    delay = float(raw)
                except (TypeError, ValueError):
                    # RFC 9110: saniye YA DA HTTP tarihi. Tarih biçimi eskiden 30 sn'ye
                    # düşüyordu → sunucunun istediğinden erken istek (Tur-78, R3).
                    try:
                        delay = parsedate_to_datetime(raw).timestamp() - time.time()
                    except (TypeError, ValueError, IndexError, OverflowError):
                        delay = 30.0
                _retry_after = time.monotonic() + max(30.0, min(delay, 3600.0))
        r.raise_for_status()
        return r.json() if as_json else r.text
    except Exception as e:
        logger.debug(f"[insider] GET başarısız {url}: {e}")
        return None


def _load_cik_map() -> dict[str, int]:
    """ticker (upper) → CIK (int) eşlemesi. Önbellekli (bellek + disk)."""
    global _cik_map
    if _cik_map is not None:
        return _cik_map
    disk = INSIDER_DIR / "_cik_map.json"
    if disk.exists():
        try:
            _cik_map = {k: int(v) for k, v in json.loads(disk.read_text("utf-8")).items()}
            return _cik_map
        except Exception:
            pass
    data = _get(CIK_MAP_URL)
    out: dict[str, int] = {}
    if isinstance(data, dict):
        for row in data.values():
            try:
                out[str(row["ticker"]).upper()] = int(row["cik_str"])
            except Exception:
                continue
    if out:
        try:
            disk.write_text(json.dumps(out), "utf-8")
        except Exception:
            pass
    _cik_map = out
    return out


def ticker_to_cik(symbol: str) -> int | None:
    # EDGAR ticker'ları '.' yerine '-' kullanır (BRK.B → BRK-B); ikisini de dene.
    m = _load_cik_map()
    s = symbol.upper()
    return m.get(s) or m.get(s.replace(".", "-")) or m.get(s.replace("-", "."))


def _iter_form4(cik: int) -> list[tuple[str, str, str]]:
    """CIK'in TÜM Form 4 dosyalamaları → [(filingDate, accession, primaryDocument), ...].

    submissions JSON 'recent' bloğu (~son 1 yıl/1000 dosya) + 'files' arşiv parçaları."""
    out: list[tuple[str, str, str]] = []

    def _collect(block: dict) -> None:
        forms = block.get("form", [])
        dates = block.get("filingDate", [])
        accs = block.get("accessionNumber", [])
        docs = block.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if form != "4":
                continue
            try:
                out.append((dates[i], accs[i], docs[i] if i < len(docs) else ""))
            except Exception:
                continue

    root = _get(SUBMISSIONS_URL.format(cik=cik))
    if not isinstance(root, dict):
        return out
    filings = root.get("filings", {})
    if isinstance(filings.get("recent"), dict):
        _collect(filings["recent"])
    for extra in filings.get("files", []) or []:
        name = extra.get("name")
        if not name:
            continue
        block = _get(f"https://data.sec.gov/submissions/{name}")
        if isinstance(block, dict):
            _collect(block)
    return out


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_form4(cik: int, accession: str, primary_doc: str) -> list[dict]:
    """Bir Form 4 XML'inden açık-piyasa (P/S) non-derivative işlemlerini çıkar.

    Dönen her kayıt: {code, shares, price, ad}  (ad = A/D acquired/disposed)."""
    acc_nodash = accession.replace("-", "")
    # primaryDocument genelde XSLT-render edilmiş HTML yolu (ör. 'xslF345X06/form4.xml')
    # → bu HTML'de ham <nonDerivativeTransaction> YOK. Ham XML aynı dosya adıdır ama
    # 'xslF345X06/' öneki olmadan. Önce ham XML'i dene, sonra primaryDocument, en son
    # tam SGML .txt (büyük indirme → yalnız son çare).
    candidates: list[str] = []
    if primary_doc:
        base_name = primary_doc.split("/")[-1]   # xsl önekini at → ham XML
        candidates.append(base_name)
        if primary_doc != base_name:
            candidates.append(primary_doc)
    candidates.append("form4.xml")               # yaygın varsayılan ham XML adı
    candidates.append(f"{accession}.txt")        # fallback: tam dosyalama (içinde XML var)
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"

    text = None
    for doc in candidates:
        if not doc:
            continue
        t = _get(f"{base}/{doc}", as_json=False)
        if t and ("<nonDerivativeTransaction" in t or "ownershipDocument" in t):
            text = t
            break
    if not text:
        return []

    # .txt fallback'te XML, <XML>...</XML> SGML zarfı içinde olabilir → ayıkla.
    if "<ownershipDocument" in text and not text.lstrip().startswith("<?xml"):
        i = text.find("<ownershipDocument")
        j = text.find("</ownershipDocument>")
        if i != -1 and j != -1:
            text = text[i : j + len("</ownershipDocument>")]

    try:
        rootx = ET.fromstring(text)
    except Exception:
        return []

    def _find(el, name):
        for c in el.iter():
            if _localname(c.tag) == name:
                return c
        return None

    def _val(el, name):
        node = _find(el, name)
        if node is None:
            return None
        v = _find(node, "value")
        return (v.text if v is not None else node.text)

    rows: list[dict] = []
    for tx in rootx.iter():
        if _localname(tx.tag) != "nonDerivativeTransaction":
            continue
        coding = _find(tx, "transactionCoding")
        code = _val(coding, "transactionCode") if coding is not None else None
        if code not in ("P", "S"):     # yalnız açık-piyasa alım/satım
            continue
        amounts = _find(tx, "transactionAmounts")
        shares = price = None
        ad = None
        if amounts is not None:
            try:
                shares = float(_val(amounts, "transactionShares") or 0)
            except (TypeError, ValueError):
                shares = None
            try:
                price = float(_val(amounts, "transactionPricePerShare") or 0)
            except (TypeError, ValueError):
                price = None
            ad = _val(amounts, "transactionAcquiredDisposedCode")
        rows.append({"code": code, "shares": shares or 0.0,
                     "price": price or 0.0, "ad": ad})
    return rows


def _build_events(symbol: str) -> pd.DataFrame | None:
    """Sembolün tüm Form 4'lerini ayrıştırıp FİLİNG tarihine göre toplulaştır.

    Sütunlar: buy_shares, sell_shares, buy_value, n_buy_filings, n_sell_filings.
    n_*_filings = o gün ilgili yönde işlem içeren AYRI dosyalama (≈ ayrı insider) sayısı
    → 'cluster' (küme alımı) ölçütü."""
    cik = ticker_to_cik(symbol)
    if cik is None:
        return None
    filings = _iter_form4(cik)
    if not filings:
        return None
    agg: dict[str, dict] = {}
    for fdate, acc, doc in filings:
        rows = _parse_form4(cik, acc, doc)
        if not rows:
            continue
        bsh = sum(r["shares"] for r in rows if r["code"] == "P")
        ssh = sum(r["shares"] for r in rows if r["code"] == "S")
        bval = sum(r["shares"] * r["price"] for r in rows if r["code"] == "P")
        d = agg.setdefault(fdate, {"buy_shares": 0.0, "sell_shares": 0.0,
                                   "buy_value": 0.0, "n_buy_filings": 0,
                                   "n_sell_filings": 0})
        d["buy_shares"] += bsh
        d["sell_shares"] += ssh
        d["buy_value"] += bval
        if bsh > 0:
            d["n_buy_filings"] += 1
        if ssh > 0:
            d["n_sell_filings"] += 1
    if not agg:
        return None
    idx = pd.to_datetime(list(agg.keys()))
    df = pd.DataFrame(list(agg.values()), index=[d.date() for d in idx])
    return df.sort_index()


def insider_events(symbol: str, *, refresh: bool = False) -> pd.DataFrame | None:
    """Sembolün insider olay tablosu (filing-tarihli, cache'li: bellek + disk).

    İndeks = datetime.date (filing günü). None → CIK yok / Form 4 yok / ağ hatası."""
    s = symbol.upper()
    if not refresh and s in _events_cache:
        return _events_cache[s]
    p = INSIDER_DIR / f"{s}.json"
    if not refresh and p.exists():
        try:
            raw = json.loads(p.read_text("utf-8"))
            ev = raw.get("events", {})
            if ev:
                df = pd.DataFrame.from_dict(ev, orient="index")
                df.index = [pd.Timestamp(i).date() for i in df.index]
                df = df.sort_index()
                _events_cache[s] = df
                return df
            _events_cache[s] = None
            return None
        except Exception:
            pass
    df = _build_events(s)
    try:
        payload = {"as_of": time.strftime("%Y-%m-%d"),
                   "events": {} if df is None else
                   {str(i): r for i, r in df.to_dict(orient="index").items()}}
        p.write_text(json.dumps(payload), "utf-8")
    except Exception:
        pass
    _events_cache[s] = df
    return df


def insider_signal(events: pd.DataFrame | None, prev_date,
                   *, min_buyers: int = 1, allow_short: bool = False
                   ) -> tuple[str, list[str]]:
    """SAF karar: önceki bar (prev_date) bir insider KÜME-ALIM dosyalama günü müydü?

    Dönüş: ('LONG'|'SHORT'|'WAIT', reasons). allow_short=True ise küme-SATIM → SHORT
    (insider satışları gürültülüdür → varsayılan KAPALI). min_buyers = aynı gün gereken
    ayrı alıcı/dosyalama sayısı (küme eşiği)."""
    if events is None or events.empty:
        return "WAIT", ["NO_INSIDER_DATA"]
    if prev_date not in events.index:
        return "WAIT", ["NO_INSIDER_EVENT"]
    row = events.loc[prev_date]
    if isinstance(row, pd.DataFrame):       # aynı tarih birden çok kez (olmamalı ama güvenli)
        row = row.iloc[-1]
    n_buy = int(row.get("n_buy_filings", 0))
    n_sell = int(row.get("n_sell_filings", 0))
    buy_sh = float(row.get("buy_shares", 0.0))
    sell_sh = float(row.get("sell_shares", 0.0))
    if n_buy >= min_buyers and buy_sh > sell_sh:
        return "LONG", [f"INSIDER_BUY_cluster_{n_buy}", f"filed_{prev_date}"]
    if allow_short and n_sell >= min_buyers and sell_sh > buy_sh:
        return "SHORT", [f"INSIDER_SELL_cluster_{n_sell}", f"filed_{prev_date}"]
    return "WAIT", [f"INSIDER_NO_CLUSTER_buy{n_buy}_sell{n_sell}"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SEC EDGAR insider (Form 4) olay tablosu")
    ap.add_argument("symbol")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ev = insider_events(args.symbol, refresh=args.refresh)
    if ev is None or ev.empty:
        print(f"{args.symbol}: insider olayı bulunamadı (CIK/Form 4/ağ).")
    else:
        buys = ev[ev["n_buy_filings"] > 0]
        print(f"{args.symbol}: {len(ev)} olay günü, {len(buys)} küme-alım günü "
              f"(CIK={ticker_to_cik(args.symbol)})")
        print(buys.tail(15).to_string())
