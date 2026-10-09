"""openFDA onay geçmişi + SEC dosyalamalarından İLERİ PDUFA takvimi. KARAR-DIŞI.

`calendar`: openFDA onay geçmişi. Sponsor adı FDA'daki şirket adıdır; borsa sembolüyle
otomatik eşleme yapılmaz. Güncel API geçmiş revizyonların yayımlanma zamanını kanıtlamaz.

`pdufa_calendar` (Tur-78): openFDA gelecekteki PDUFA tarihini TAŞIMAZ, ama şirketler
onları 8-K/6-K basın bültenlerinde duyurur. SEC EDGAR tam-metin araması ANAHTARSIZDIR
(ölçüldü 2026-09-14: 3 ayda 95 dosyalama, 74'ünde tarih, 43 benzersiz ileri tarih,
sembol eşleme %100). Tarih ŞİRKET BEYANIDIR, FDA takvimi değildir; uzatma/erken onay
ancak sonraki dosyalamada görünür. Bilinme tarihi dosyalama günüdür (`file_date <= as_of`).
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import re
import time
import urllib.parse
from datetime import UTC, date, datetime, timedelta

from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR
from nihai.net import contact_ua, shared_client

CACHE_DIR = DATA_DIR / "fda_cache"
API_URL = "https://api.fda.gov/drug/drugsfda.json"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/{doc}"

_AY = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_TARIH = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
                    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2}),?\s+(20\d\d)\b", re.I)
_KABA = re.compile(r"\b(?:(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?(20\d\d)"
                   r"|q([1-4])\s*(20\d\d)|(?:first|second)\s+half\s+(?:of\s+)?(20\d\d)|(?:in|by)\s+(20\d\d))\b", re.I)
# Şirket kısaltmaları cümleyi bölmez; büyük harf şartı yalnız cümle başlangıcındadır.
_CUMLE_SONU = re.compile(r"[;!?]|(?<!(?i:\bInc))(?<!(?i:\bCorp))(?<!(?i:\bLtd))(?<!(?i:\bCo))"
                         r"(?<!(?i:\bplc))(?<!(?i:\bN\.V))(?<!(?i:\bS\.A))\.\s+[A-Z(]|\b[ap]\.m\.")
_DATE = re.compile(r"\bdates?\b", re.I)
# Tarihe BİTİŞİK etkinlik ("conference on <tarih>") o tarihi etkinliğe bağlar. Arada metnin herhangi
# bir yerindeki sözcük reddetmez: "in the event of a major amendment, may be extended to <tarih>" ve
# "following a Type A meeting with the FDA, is <tarih>" gerçek PDUFA tarihidir (Tur-78 inceleme).
# ponytail: sözcük filtresi; gerçek hedefleri elerse olay-tarih ilişkisini ayrıştır.
_ETKINLIK = re.compile(r"\b(?:conference|call|webcast|meeting|presentation|event|discussed)(?:\s+\w+){0,2}\s*$",
                       re.I)
_CEYREK = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
_TICKER = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,7})(?:,\s*[A-Z0-9.\-]+)*\)")


def approval_events(payload: dict, since: str, until: str) -> list[dict] | None:
    """Her başvurunun kendi durum tarihini süzer; üst kayıt eşleşmesi yeterli değildir."""
    try:
        lo, hi = date.fromisoformat(since), date.fromisoformat(until)
        records = payload["results"]
        if not isinstance(records, list) or lo > hi:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    events: dict[tuple, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        # Tur-78 (R3-B7): liste olmayan alan (ör. NaN) kaydı atlatır/boş sayılır, istisna değil.
        subs, prods = record.get("submissions"), record.get("products")
        prods = prods if isinstance(prods, list) else []
        for submission in subs if isinstance(subs, list) else []:
            if not isinstance(submission, dict) or submission.get("submission_status") != "AP":
                continue
            try:
                day = datetime.strptime(str(submission["submission_status_date"]), "%Y%m%d").date()
            except (KeyError, ValueError):
                continue
            if not lo <= day <= hi:
                continue
            app = record.get("application_number")
            if not isinstance(app, str) or not app:
                continue
            kind, number = submission.get("submission_type"), submission.get("submission_number")
            if not isinstance(kind, str) or not isinstance(number, str):
                continue
            key = (app, kind, number, day.isoformat())
            events[key] = dict(application=app, sponsor=record.get("sponsor_name"), date=day.isoformat(),
                submission_type=key[1], submission_number=key[2], status="AP",
                products=sorted({p["brand_name"] for p in prods
                                 if isinstance(p, dict) and isinstance(p.get("brand_name"), str)}))
    return sorted(events.values(), key=lambda row: (row["date"], row["application"]), reverse=True)


def calendar(sponsor: str, *, since: str = "2022-01-01", until: str | None = None,
             fetch: bool = False, force: bool = False) -> dict | None:
    """Tek sponsor sorgusu, atomik damıtılmış önbellek. fetch=False hiçbir zaman ağ açmaz."""
    try:
        sponsor = sponsor.strip()
        lo, hi = date.fromisoformat(since), min(date.fromisoformat(until) if until else date.today(), date.today())
        if not re.fullmatch(r"[\w .,&'()-]{1,120}", sponsor) or lo > hi:
            return None
    except (TypeError, ValueError, AttributeError):
        return None
    key = hashlib.sha256(f"{sponsor.upper()}|{lo}|{hi}".encode()).hexdigest()
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
        query = f'sponsor_name:{json.dumps(sponsor)} AND submissions.submission_status_date:[{lo:%Y%m%d} TO {hi:%Y%m%d}]'
        response = shared_client().get(API_URL, params={"search": query, "limit": 1000},
                                       headers={"User-Agent": contact_ua("NihaiFDA/1.0")}, timeout=60)
        payload = response.json()
        if response.status_code == 404 and payload.get("error", {}).get("code") == "NOT_FOUND":
            payload = {"results": [], "meta": {"results": {"total": 0}}}
        else:
            response.raise_for_status()
        events = approval_events(payload, lo.isoformat(), hi.isoformat())
        if events is None:
            raise ValueError("FDA yanıt şeması geçersiz")
        total = int(payload["meta"]["results"]["total"])
        result = dict(ok=True, report_type="fda_calendar", sponsor=sponsor, since=lo.isoformat(), until=hi.isoformat(),
            fetched_at=datetime.now(UTC).isoformat(), events=events, n_applications=len(payload["results"]),
            total_applications=total, truncated=total > len(payload["results"]), upcoming_pdufa=None,
            source=API_URL, pit_scope="status_date_not_publication_vintage",
            note="Onay geçmişidir. Gelecekteki PDUFA tarihleri bu kaynakta yok; sponsor borsa sembolüne otomatik eşlenmez.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(result, ensure_ascii=False, allow_nan=False))
        return {**result, "cache_state": "downloaded"}
    except Exception as exc:
        logger.warning(f"[fda] veri yenilenemedi: {type(exc).__name__}")
        return {**cached, "cache_state": "stale", "refresh_failed": True} if isinstance(cached, dict) else None


def _tarih(tok_ay: str, gun: str, yil: str) -> date | None:
    try:
        return date(int(yil), _AY.index(tok_ay[:3].lower()) + 1, int(gun))
    except ValueError:
        return None


def pdufa_mentions(text: str) -> list[dict]:
    """Her 'PDUFA' geçişinden SONRAKİ ilk tarih: gün, çeyrek, yarıyıl veya yıl hassasiyeti.

    Kabul: aradaki bağlaç 'date/dates' sözcüğü içerir, 160 karakteri aşmaz; cümle sonu
    veya etkinlik sözcüğü taşımaz. Şirket kısaltmaları cümle sonu sayılmaz. Ölçülen
    yanlış pozitif ("Target PDUFA Date in 2028 Conference Call ... 8:30 a.m. ET ... Pa.,
    August 4, 2026") böylece yıl hassasiyetinde okunur, bülten tarih satırı alınmaz.
    "from <tarih> to <tarih>" kalıbında YENİ tarih alınır ve `revised=True` olur.
    """
    if not isinstance(text, str) or not text:
        return []
    plain = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", text)))
    out: list[dict] = []
    seen: set[tuple] = set()
    for m in re.finditer(r"PDUFA", plain):
        win = plain[m.end(): m.end() + 220]
        adaylar = [x for x in (_TARIH.search(win), _KABA.search(win)) if x]
        if not adaylar:
            continue
        ilk = min(adaylar, key=lambda x: x.start())
        arada = win[: ilk.start()]
        if (len(arada) > 160 or not _DATE.search(arada)
                or _CUMLE_SONU.search(arada) or _ETKINLIK.search(arada)):
            continue
        revised = False
        row: dict[str, str | None] | None = None
        if ilk.re is _TARIH:
            gun = _tarih(*ilk.groups())
            baglac = re.match(r"\s*to\s+", win[ilk.end():], re.I)
            sonra = _TARIH.match(win, ilk.end() + baglac.end()) if baglac else None
            if sonra is not None and arada.rstrip().lower().endswith("from"):
                gun, revised = _tarih(*sonra.groups()), True
            if gun is not None:
                row = {"pdufa_date": gun.isoformat(), "precision": "day", "period": gun.isoformat()}
        else:
            q, y, qn, qy, hy, yy = ilk.groups()
            if y:
                row = {"precision": "quarter", "period": f"{y}-Q{_CEYREK[q.lower()]}"}
            elif qy:
                row = {"precision": "quarter", "period": f"{qy}-Q{qn}"}
            elif hy:
                row = {"precision": "half", "period": f"{hy}-H{1 if ilk.group(0).lower().startswith('first') else 2}"}
            else:
                row = {"precision": "year", "period": yy}
            row["pdufa_date"] = None
        if row is None or (row["precision"], row["period"]) in seen:
            continue
        seen.add((row["precision"], row["period"]))
        out.append({**row, "revised": revised or bool(re.search(r"\b(extend|new PDUFA|revised)", arada, re.I)),
                    "context": plain[max(0, m.start() - 100): m.end() + ilk.end() + 30][:320]})
    return out


def _efts_hits(since: str, until: str, *, max_filings: int) -> tuple[list[dict], int] | None:
    """EFTS tam-metin araması (8-K, 6-K); SEC kuyruğu `insider._get` ile paylaşılır."""
    from nihai.data.insider import _get

    hits: list[dict] = []
    total = 0
    while len(hits) < max_filings:
        query = urllib.parse.urlencode({"q": '"PDUFA"', "forms": "8-K,6-K", "dateRange": "custom",
                                        "startdt": since, "enddt": until, "from": len(hits)})
        page = _get(f"{EFTS_URL}?{query}")
        if not isinstance(page, dict):
            return None if not hits else (hits, total)
        block = page.get("hits") or {}
        rows = block.get("hits") or []
        total = int((block.get("total") or {}).get("value") or 0)
        hits.extend(r for r in rows if isinstance(r, dict))
        if not rows or len(hits) >= min(total, 10000):
            break
    return hits[:max_filings], total


def _upcoming(mentions: list[dict], ref: date) -> dict:
    """as_of anında BİLİNEN (file_date <= as_of) ve günü gelmemiş gün-hassasiyetli tarihler.

    (sembol, tarih) tekildir. Aynı sembolde sonraki bir dosyalama `revised` yeni tarih
    bildirmişse eski tarih `possibly_superseded` işaretlenir (ilaç adı eşlenmez)."""
    ref_s = ref.isoformat()
    known = [m for m in mentions if isinstance(m, dict) and str(m.get("file_date", "9999")) <= ref_s]
    rows: dict[tuple, dict] = {}
    for m in known:
        if m.get("precision") != "day" or not m.get("pdufa_date") or m["pdufa_date"] < ref_s:
            continue
        key = (m.get("ticker"), m["pdufa_date"])
        r = rows.setdefault(key, {"ticker": key[0], "company": m.get("company"), "pdufa_date": key[1],
                                  "first_seen": m["file_date"], "last_seen": m["file_date"], "n_mentions": 0,
                                  "revised": False, "context": m.get("context")})
        r["n_mentions"] += 1
        r["first_seen"] = min(r["first_seen"], m["file_date"])
        if m["file_date"] >= r["last_seen"]:
            r["last_seen"], r["context"] = m["file_date"], m.get("context")
        r["revised"] = r["revised"] or bool(m.get("revised"))
    for r in rows.values():
        r["possibly_superseded"] = any(
            m.get("ticker") == r["ticker"] and m.get("revised") and m.get("pdufa_date") not in (None, r["pdufa_date"])
            and m["file_date"] > r["last_seen"] for m in known)
    up = sorted(rows.values(), key=lambda r: (r["pdufa_date"], r["ticker"] or ""))
    return {"as_of": ref_s, "upcoming": up, "n_upcoming": len(up)}


def pdufa_calendar(*, since: str | None = None, until: str | None = None, as_of: str | None = None,
                   fetch: bool = False, force: bool = False, max_filings: int = 1500) -> dict | None:
    """SEC 8-K/6-K bültenlerinden PDUFA takvimi; fetch=False hiçbir zaman ağ açmaz.

    Varsayılan pencere `until`dan 400 gün geriye (standart inceleme ~12 ay önce duyurulur).
    Önbellek bahisleri tutar; `upcoming` her okumada `as_of`a göre yeniden süzülür."""
    try:
        today = date.today()
        hi = min(date.fromisoformat(until) if until else today, today)
        lo = date.fromisoformat(since) if since else hi - timedelta(days=400)
        ref = date.fromisoformat(as_of) if as_of else hi
        if lo > hi or not 1 <= int(max_filings) <= 10000:
            return None
    except (TypeError, ValueError):
        return None
    path = CACHE_DIR / f"pdufa_{lo}_{hi}.json"
    cached = None
    age = float("inf")
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - path.stat().st_mtime
    except (OSError, ValueError):
        pass
    if not isinstance(cached, dict) or not isinstance(cached.get("mentions"), list):
        cached = None
    if not fetch or (cached is not None and age < 86400 and not force):
        return (None if cached is None else
                {**cached, **_upcoming(cached["mentions"], ref), "cache_state": "fresh" if age < 86400 else "stale"})
    try:
        from nihai.data.insider import _get

        got = _efts_hits(lo.isoformat(), hi.isoformat(), max_filings=int(max_filings))
        if got is None:
            raise ValueError("EFTS yanıtı yok")
        hits, total = got
        mentions: list[dict] = []
        n_read = n_failed = n_dated = 0
        for hit in hits:
            src = hit.get("_source") or {}
            try:
                adsh, doc = str(hit["_id"]).split(":", 1)
                cik = str(int(src["ciks"][0]))
                file_date = date.fromisoformat(str(src["file_date"])).isoformat()
            except (KeyError, IndexError, TypeError, ValueError):
                n_failed += 1
                continue
            text = _get(ARCHIVE_URL.format(cik=cik, adsh=adsh.replace("-", ""), doc=doc), as_json=False, timeout=30)
            if not isinstance(text, str):
                n_failed += 1
                continue
            n_read += 1
            found = pdufa_mentions(text)
            n_dated += bool(found)
            name = str((src.get("display_names") or [""])[0])
            ticker = _TICKER.search(name)
            for f in found:
                mentions.append({**f, "ticker": ticker.group(1) if ticker else None,
                                 "company": re.sub(r"\s*\(.*$", "", name).strip() or None, "cik": cik,
                                 "form": src.get("form"), "file_date": file_date, "accession": adsh, "document": doc})
        if hits and not n_read:
            raise ValueError("hiçbir dosyalama okunamadı")
        result = dict(ok=True, report_type="pdufa_calendar", since=lo.isoformat(), until=hi.isoformat(),
                      fetched_at=datetime.now(UTC).isoformat(), n_filings_total=total, n_filings_read=n_read,
                      n_filings_failed=n_failed, n_filings_with_date=n_dated, truncated=total > len(hits),
                      mentions=mentions, source=EFTS_URL, pit_scope="file_date_is_known_date",
                      note="Şirket beyanıdır, FDA takvimi değildir; uzatma/erken onay sonraki dosyalamada görünür. "
                           "EFTS indeksi sonradan güncellenebilir.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(result, ensure_ascii=False, allow_nan=False))
        return {**result, **_upcoming(mentions, ref), "cache_state": "downloaded"}
    except Exception as exc:
        logger.warning(f"[fda] PDUFA takvimi yenilenemedi: {type(exc).__name__}")
        return (None if cached is None else
                {**cached, **_upcoming(cached["mentions"], ref), "cache_state": "stale", "refresh_failed": True})


def _cli() -> None:
    import argparse

    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="FDA sponsor onay geçmişi (--sponsor) veya SEC 8-K PDUFA takvimi (--pdufa)")
    ap.add_argument("--sponsor")
    ap.add_argument("--pdufa", action="store_true")
    ap.add_argument("--since", default=None)
    ap.add_argument("--until", default=None)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.pdufa:
        result = pdufa_calendar(since=args.since, until=args.until, as_of=args.as_of,
                                fetch=not args.no_fetch, force=args.force)
    elif args.sponsor:
        result = calendar(args.sponsor, since=args.since or "2022-01-01", until=args.until,
                          fetch=not args.no_fetch, force=args.force)
    else:
        ap.error("--sponsor veya --pdufa gerekli")
    print(json.dumps(result if result is not None else {"ok": False, "reason": "FDA ölçümü yok"}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
