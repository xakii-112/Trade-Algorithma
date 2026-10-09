"""
etf_holdings.py — SPDR ETF günlük sepet ağırlıkları + örtük korelasyon serisi (Tur-79). KARAR-DIŞI.

NEDEN: yol haritası §7.5 "dispersiyon ticareti / örtük korelasyon". Çekirdek kimlik
(`options_math.implied_correlation`) Tur-73'ten beri vardı ama tek bir anlık görüntü içindi ve
Tur-72 açık kalemi *"N bileşen zinciri çekmek uç başına N ağ turu"* diye ertelenmişti. İki parça
eksikti ve ikisi de ANAHTARSIZ bulundu:

  1. Bileşen örtük vol'leri — DoltHub `volatility_history` toplu kopyası ZATEN diskte
     (`iv_current`, 2019-02 → , 2321 sembol; Tur-64 parquet yan dosyası). Ağ turu YOK.
  2. Endeks ağırlıkları — State Street SPDR günlük sepet dosyası (xlsx):
     `https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-<etf>.xlsx`
     Ölçüldü (2026-09-16): SPY 505 satır, ağırlık toplamı %99,95, "As of 14-Sep-2026".
     `openpyxl` KURULU DEĞİL → xlsx stdlib `zipfile` + `xml.etree` ile okunur (yeni bağımlılık yok).

DIŞ DOĞRULAMA (ölçüldü 2026-09-16, 950 ortak gün, 2021-01 → 2026-08): bu seri Cboe'nin resmî
COR1M endeksiyle (ilk 50 bileşen, ATM örtük vol, aynı kimlik) seviyede **0,945**, GÜNLÜK
DEĞİŞİMDE **0,893** korelasyonlu; 2026'da ortalamalar 0,132 ↔ 0,131. Erken yıllarda seviye
AŞAĞI yanlıdır (2022: 0,287 ↔ 0,419) ve nedeni bilinmektedir: ağırlıklar BUGÜNÜN sepetidir
(`agirlik_sabit=True`) — 2021'in ilk 50'si bugünkünden farklıydı. Bu bir PIT ağırlık açığıdır,
seri geçmiş için ÖLÇÜLMÜŞ bir "o günkü ρ" DEĞİLDİR; yalnız son dönemde ve değişim yönünde güvenilir.

SÖZLEŞME: `fetch=False` → önbellek-yalnız, ağa HİÇ çıkmaz (HTTP ucundan çağıranlar bunu
kullanır — BULGU-59-8). Önbelleği doldurmak ayrı adımdır: `python -m nihai.data.etf_holdings --warm SPY`.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

CACHE_DIR = DATA_DIR / "etf_holdings_cache"
SSGA_URL = ("https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/"
            "etfs/us/holdings-daily-us-en-{etf}.xlsx")
MAX_AGE_DAYS = 7
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_AYLAR = {a: i for i, a in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_KOLONLAR = ["ticker", "name", "weight", "shares", "cusip", "sector"]


def _bos() -> pd.DataFrame:
    return pd.DataFrame(columns=_KOLONLAR)


def _as_of(metin: str) -> str | None:
    """'As of 14-Sep-2026' → '2026-09-14'. Yerel ayara bağlı `%b` KULLANILMAZ (Türkçe yerel
    ayarda 'Sep' tanınmaz ve sessizce None olurdu) — ay adı sabit tablodan çözülür."""
    m = re.search(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", metin or "")
    if not m or m.group(2).lower() not in _AYLAR:
        return None
    try:
        return datetime(int(m.group(3)), _AYLAR[m.group(2).lower()], int(m.group(1))).date().isoformat()
    except ValueError:
        return None


def parse_xlsx(raw: bytes) -> pd.DataFrame | None:
    """SSGA sepet xlsx'ini çerçeveye çevir: [ticker, name, weight (ONDALIK), shares, cusip, sector].

    Başlık satırı ('Name', 'Ticker', …) içerikten bulunur (üstte 3-4 satırlık fon künyesi vardır);
    ağırlık dosyada YÜZDEdir → 100'e bölünür. Tarih `attrs['as_of']`da. Bozuk dosya → None."""
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        paylasilan: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            kok = ET.fromstring(z.read("xl/sharedStrings.xml"))
            paylasilan = ["".join(t.text or "" for t in si.iter(_NS + "t")) for si in kok.findall(_NS + "si")]
        sayfa = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError, ValueError):
        return None
    satirlar: list[list[str | None]] = []
    for r in sayfa.iter(_NS + "row"):
        degerler: list[str | None] = []
        for c in r.findall(_NS + "c"):
            v = c.find(_NS + "v")
            tip = c.get("t")
            if tip == "s" and v is not None and v.text is not None:
                try:
                    degerler.append(paylasilan[int(v.text)])
                except (ValueError, IndexError):
                    degerler.append(None)
            elif tip == "inlineStr":
                degerler.append("".join(t.text or "" for t in c.iter(_NS + "t")))
            else:
                degerler.append(v.text if v is not None else None)
        satirlar.append(degerler)
    as_of = next((_as_of(str(s[1])) for s in satirlar[:10]
                  if len(s) > 1 and s[0] and str(s[0]).startswith("Holdings")), None)
    try:
        bas = next(i for i, s in enumerate(satirlar) if s and s[0] == "Name")
    except StopIteration:
        return None
    baslik = [str(x or "").strip().lower() for x in satirlar[bas]]
    gerek = {"name", "ticker", "weight"}
    if not gerek.issubset(baslik):
        return None
    ix = {ad: baslik.index(ad) for ad in baslik if ad}
    kayit = []
    for s in satirlar[bas + 1:]:
        def al(ad: str, s: list[str | None] = s) -> str | None:
            i = ix.get(ad)
            return s[i] if i is not None and i < len(s) else None
        tk = al("ticker")
        if not tk or not str(tk).strip() or str(tk).strip() == "-":
            continue
        kayit.append({"ticker": str(tk).strip().upper(), "name": al("name"), "weight": al("weight"),
                      "shares": al("shares held"), "cusip": al("identifier"), "sector": al("sector")})
    if not kayit:
        return None
    d = pd.DataFrame(kayit, columns=_KOLONLAR)
    d["weight"] = pd.to_numeric(d["weight"], errors="coerce") / 100.0
    d["shares"] = pd.to_numeric(d["shares"], errors="coerce")
    d = d.dropna(subset=["weight"])
    d = d[d["weight"] > 0].sort_values("weight", ascending=False, kind="mergesort").reset_index(drop=True)
    if d.empty:
        return None
    d.attrs["as_of"] = as_of
    return d


def _yollar(etf: str) -> tuple[Path, Path]:
    e = etf.strip().upper()
    return CACHE_DIR / f"{e}.csv", CACHE_DIR / f"{e}.meta.json"


def holdings(etf: str = "SPY", *, fetch: bool = False, max_age_days: int = MAX_AGE_DAYS) -> pd.DataFrame:
    """ETF sepeti (ağırlığa göre azalan). `attrs`: `as_of` (dosyanın kendi tarihi), `kaynak`
    ('onbellek' | 'indirildi' | 'bayat_onbellek'), `bayat_gun` (indirme yaşı, gün).

    `fetch=False` → önbellek-yalnız (bayat da olsa döner, bayatlığı `bayat_gun` taşır); önbellek
    yoksa BOŞ çerçeve. `fetch=True` → önbellek `max_age_days`ten tazeyse ağa çıkmaz; indirme
    başarısızsa bayat önbelleğe düşer (veri > veri yok) ve bunu `kaynak`ta söyler."""
    csv_yol, meta_yol = _yollar(etf)

    def oku(kaynak: str) -> pd.DataFrame | None:
        if not csv_yol.exists():
            return None
        try:
            d = pd.read_csv(csv_yol, dtype={"ticker": str, "cusip": str})
            meta = json.loads(meta_yol.read_text(encoding="utf-8")) if meta_yol.exists() else {}
        except (OSError, ValueError) as e:
            logger.debug(f"[etf_holdings] önbellek okunamadı {csv_yol.name}: {e}")
            return None
        indirme = meta.get("indirildi")
        yas = None
        if indirme:
            try:
                yas = (datetime.now() - datetime.fromisoformat(indirme)).days
            except ValueError:
                yas = None
        d.attrs.update({"as_of": meta.get("as_of"), "kaynak": kaynak, "bayat_gun": yas, "etf": etf.upper()})
        return d

    onbellek = oku("onbellek")
    if not fetch:
        return onbellek if onbellek is not None else _bos()
    if onbellek is not None and onbellek.attrs.get("bayat_gun") is not None \
            and onbellek.attrs["bayat_gun"] < max_age_days:
        return onbellek
    try:
        from nihai.net import shared_client
        r = shared_client().get(SSGA_URL.format(etf=etf.strip().lower()), timeout=60.0,
                                headers={"User-Agent": "Mozilla/5.0 (NihaiAlgoritma research)"},
                                follow_redirects=True)
        r.raise_for_status()
        d = parse_xlsx(r.content)
    except Exception as e:  # noqa: BLE001 — veri katmanı fail-soft
        logger.warning(f"[etf_holdings] {etf} indirilemedi: {e}")
        d = None
    if d is None:
        bayat = oku("bayat_onbellek")
        return bayat if bayat is not None else _bos()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    as_of = d.attrs.get("as_of")
    atomic_write_text(csv_yol, d.to_csv(index=False))
    atomic_write_text(meta_yol, json.dumps({"as_of": as_of, "indirildi": datetime.now().isoformat(timespec="seconds"),
                                            "satir": int(len(d))}, ensure_ascii=False))
    d.attrs.update({"as_of": as_of, "kaynak": "indirildi", "bayat_gun": 0, "etf": etf.upper()})
    return d


def implied_correlation_frame(etf: str = "SPY", n_top: int = 50, *, fetch: bool = False,
                              min_kapsama: float = 0.9, iv_path: str | Path | None = None,
                              holdings_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Günlük örtük korelasyon serisi: ETF örtük vol'ü ↔ ilk `n_top` bileşenin örtük vol'leri.

    Cboe COR1M ile AYNI kimlik ve AYNI evren tanımı (ilk 50, ATM örtük vol); vol kaynağı DoltHub
    `iv_current`, ağırlık kaynağı SSGA sepeti (`holdings`). Matematik tek kaynaktan:
    `options_math.implied_correlation_series` (kapsama kapısı dahil).

    Dönüş: index=tarih; kolonlar `rho_imp`, `kapsama`, `n`, `gecerli`. `attrs`: `as_of` (ağırlık
    tarihi), `agirlik_sabit=True`, `n_top`, `eksik_semboller` (dolt'ta hiç olmayan bileşenler),
    `iv_son_tarih`. Veri yoksa BOŞ çerçeve (istisna yok). `fetch` yalnız sepete uygulanır; dolt
    toplu kopyası bu fonksiyondan ASLA indirilmez (230 MB — ayrı adım: `dolt_options.bulk_csv`)."""
    from nihai.ai.options_math import implied_correlation_series
    from nihai.data import dolt_options as dolt

    h = holdings_df if holdings_df is not None else holdings(etf, fetch=fetch)
    if h is None or h.empty or n_top < 2:
        return pd.DataFrame()
    ust = h.head(int(n_top))
    agirlik = pd.Series(ust["weight"].to_numpy(dtype=float), index=ust["ticker"].astype(str).str.upper())
    agirlik = agirlik.groupby(level=0).sum()       # aynı sembol iki satırda gelirse (sınıf hisseleri) topla
    yol = Path(iv_path) if iv_path is not None else (
        dolt.DOLT_CACHE_DIR / f"volatility_history__{dolt.DOLT_BRANCH.replace('/', '_').replace('~', '-')}.csv")
    istenen = set(agirlik.index) | {etf.upper()}
    v = dolt.read_bulk(yol, cols=["date", "act_symbol", "iv_current"], symbols=istenen)
    if v.empty:
        return pd.DataFrame()
    v = v.assign(date=pd.to_datetime(v["date"], errors="coerce"),
                 act_symbol=v["act_symbol"].astype(str).str.upper()).dropna(subset=["date"])
    tablo = v.pivot_table(index="date", columns="act_symbol", values="iv_current", aggfunc="last").sort_index()
    if etf.upper() not in tablo.columns:
        return pd.DataFrame()
    eksik = sorted(set(agirlik.index) - set(tablo.columns))
    bilesen = tablo.reindex(columns=agirlik.index)
    seri = implied_correlation_series(tablo[etf.upper()].to_numpy(dtype=float),
                                      bilesen.to_numpy(dtype=float), agirlik.to_numpy(dtype=float),
                                      min_kapsama=min_kapsama)
    if seri is None:
        return pd.DataFrame()
    out = pd.DataFrame({"rho_imp": seri["rho_imp"], "kapsama": seri["kapsama"], "n": seri["n"],
                        "gecerli": seri["gecerli"]}, index=tablo.index)
    out.attrs.update({"as_of": h.attrs.get("as_of"), "agirlik_sabit": True, "n_top": int(n_top),
                      "eksik_semboller": eksik, "iv_son_tarih": str(tablo.index.max().date()),
                      "etf": etf.upper(), "min_kapsama": float(min_kapsama),
                      "gecerli_gun": int(np.sum(seri["gecerli"]))})
    return out


if __name__ == "__main__":  # pragma: no cover - canlı ağ gerektirir
    import argparse

    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="SPDR sepetleri + örtük korelasyon serisi")
    ap.add_argument("--warm", default="SPY", help="virgülle ETF listesi (önbelleği doldur)")
    ap.add_argument("--rho", action="store_true", help="SPY örtük korelasyonunun son 5 gününü bas")
    a = ap.parse_args()
    for e in [x.strip() for x in a.warm.split(",") if x.strip()]:
        d = holdings(e, fetch=True, max_age_days=0)
        print(f"{e:5} {len(d):>4} satır  as_of={d.attrs.get('as_of')}  kaynak={d.attrs.get('kaynak')}  "
              f"ağırlık toplamı={d['weight'].sum() if not d.empty else float('nan'):.4f}")
    if a.rho:
        f = implied_correlation_frame("SPY")
        print(f.tail(5).to_string() if not f.empty else "seri BOŞ (dolt toplu kopyası ya da sepet yok)")
        print({k: v for k, v in f.attrs.items() if k != "eksik_semboller"}, "eksik:", f.attrs.get("eksik_semboller"))
