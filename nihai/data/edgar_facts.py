"""
edgar_facts.py — SEC EDGAR XBRL `companyfacts` köprüsü. ANAHTAR GEREKMEZ.

Tur-75'in triyajı listenin BÖLÜM 9'unu (176-200, şirket temelleri) tek cümleyle
kapatmıştı: *"bu 23 madde 'veri yok' DEĞİL, 'köprü yazılmadı'dır"*. Köprü budur.

    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Ölçüldü (2026-09-07, bu makine): AAPL için **200 · 3,79 MB · 1,1 sn**, anahtar YOK.
Tek şart SEC'in "kendini tanıt" politikasıdır → `nihai.net.contact_ua` (`.env`deki
`SEC_USER_AGENT`; Tur-69).

## Üç ölçülmüş tasarım kararı (varsayım değil)

**1. `filed` BİLİNME TARİHİDİR, `end` değil.** Bir bilanço kalemi dönem sonunda
(`end`) OLUŞUR ama ancak dosyalandığında (`filed`) HALKA AÇIK olur; aradaki fark
10-K'da tipik 30-60 gündür. `end`e göre okuyan bir backtest geleceği okur — bu,
BULGU-75-3'ün (FRED `observation_date`) ve ACI DERS #7'nin şirket-temeli sürümüdür.
Bu modülde HER okuma `filed <= as_of` ile sınırlıdır ve `as_of=None` "bugün"dür.
Yeniden beyanlar (restatement) da bu kuralla doğru davranır: aynı `end` için birden
çok satır varsa, `as_of`ta BİLİNEN en son beyan alınır — sonradan düzeltilmiş rakam
geçmişe sızmaz.

**2. TEK ETİKET OKUYAN OKUYUCU SESSİZCE YANLIŞ SAYI ÜRETİR.** Ölçüldü (aynı gün,
üç şirket): `Revenues` AAPL'de VAR ama **11 satır** ve sonuncusu **2018**'dir; gerçek
kalem `RevenueFromContractWithCustomerExcludingAssessedTax`tır (117 satır). Yani
`Revenues` okuyan bir Piotroski skoru AAPL için 2018 hasılatını 2026 kârıyla
kıyaslardı ve HATA VERMEZDİ. Aynı ayrışma her satırda var: `CostOfRevenue` AAPL'de
YOK / TSLA'da VAR · `CostOfGoodsAndServicesSold` TSLA'da YOK / AAPL'de VAR ·
`PaymentsOfDividends` MSFT'de YOK / `PaymentsOfDividendsCommonStock` VAR. Bu yüzden
her kanonik kalem bir **ETİKET ZİNCİRİdir** (`CONCEPTS`) ve zincir **dönem başına**
uygulanır: bir dönemi zincirdeki ilk TUTAN etiket doldurur. Kalem başına tek etiket
seçmek yetmez — ölçüldü: AAPL'de `PaymentsOfDividendsCommonStock` zincirde önde ama
**2017'de biter**, `PaymentsOfDividends` 2026'ya gider; tek-etiket kuralı 2018 sonrası
tüm temettüleri NaN yapıyordu. Hangi dönemin hangi etiketten geldiği
`attrs["etiket"]`te (etiket → dönem sayısı), tanımı karışan kalemler
`attrs["karisik"]`ta durur — sessiz seçim de sessiz karışım da yok.

**3. ÖNBELLEK HAM DEĞİL, DAMITILMIŞ TUTULUR.** Ham JSON şirket başına ~3,8 MB'tır;
500 sembol ≈ 1,9 GB eder ve `data/` OneDrive ile SENKRONLANIR (CLAUDE.md'nin 10 GB
`option_chain` kuralının aynısı). Diske yalnız `CONCEPTS`in seçtiği satırlar CSV
olarak yazılır (AAPL: ~130 KB, 31 kalem × 20 mali yıl + çeyrekler).

## Bu modül KARAR VERMEZ

Sayı üretir; eşik/kural İÇERMEZ. Listenin önerdiği kurallar ("F-Score ≥ 7 ise al",
"M-Score > −1,78 ise vetola") KODA YAZILMADI: bir kural bir kenar iddiasıdır ve
`research/edge_lab` kanıt kapısından geçmeden `signals`/`risk`e giremez (ACI DERS
#3/#5). Skorlar `nihai/ai/fundamentals.py`tedir; burası yalnız veri taşır.
"""
from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

# TEK EDGAR İSTEMCİSİ: `insider` zaten kibar throttle (SEC_THROTTLE) + `contact_ua`
# üzerinden gidiyor ve CIK haritasını diskte tutuyor. İkinci bir istemci yazmak SEC'in
# 10 istek/sn sınırını İKİ AYRI sayaçla yorardı ve UA politikası iki yerde bakım
# isterdi (Tur-69'un `contact_ua` merkezileştirmesinin tersi).
from nihai.data.insider import _get as _edgar_get
from nihai.data.insider import ticker_to_cik

CACHE_DIR = DATA_DIR / "edgar_cache"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
MAX_AGE_DAYS = 30          # 10-K/10-Q üç ayda bir gelir; günlük tazeleme SEC'i boşuna yorar
YIL_GUN_MIN, YIL_GUN_MAX = 300, 400   # "yıllık" dönem penceresi (52/53 haftalık takvimler)

# Kanonik kalem → us-gaap etiket ZİNCİRİ (öncelik sırası; ilk TUTAN kazanır).
# Zincirler yukarıdaki (2) numaralı ölçümün sonucudur — tek etiket YETMEZ.
CONCEPTS: dict[str, tuple[str, ...]] = {
    # --- bilanço (anlık) ---
    "varliklar": ("Assets",),
    "donen_varliklar": ("AssetsCurrent",),
    "yukumlulukler": ("Liabilities",),
    "kisa_vadeli_yukumluluk": ("LiabilitiesCurrent",),
    "ozkaynak": ("StockholdersEquity",
                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "nakit": ("CashAndCashEquivalentsAtCarryingValue",
              "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "stok": ("InventoryNet",),
    "alacak": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),
    "borc_satici": ("AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"),
    "uzun_vadeli_borc": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "dagitilmamis_kar": ("RetainedEarningsAccumulatedDeficit",),
    "serefiye": ("Goodwill",),
    "duran_varlik": ("PropertyPlantAndEquipmentNet",),
    "kiralama_yukumlulugu": ("OperatingLeaseLiability", "OperatingLeaseLiabilityNoncurrent"),
    "emeklilik_yukumlulugu": ("DefinedBenefitPlanBenefitObligation",
                              "DefinedBenefitPlanAccumulatedBenefitObligation"),
    "emeklilik_varligi": ("DefinedBenefitPlanFairValueOfPlanAssets",),
    # --- gelir tablosu (dönemsel) ---
    "hasilat": ("RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"),
    "satis_maliyeti": ("CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"),
    "brut_kar": ("GrossProfit",),
    "faaliyet_kari": ("OperatingIncomeLoss",),
    "net_kar": ("NetIncomeLoss", "ProfitLoss"),
    "vergi_oncesi_kar": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    "faiz_gideri": ("InterestExpense", "InterestExpenseDebt", "InterestIncomeExpenseNet"),
    "arge": ("ResearchAndDevelopmentExpense",),
    "reklam": ("AdvertisingExpense", "AdvertisingAndPromotionExpense"),
    "sga": ("SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"),
    "amortisman": ("DepreciationDepletionAndAmortization",
                   "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortization"),
    "serefiye_bozulmasi": ("GoodwillImpairmentLoss", "GoodwillAndIntangibleAssetImpairment"),
    # --- nakit akışı (dönemsel) ---
    "faaliyet_nakit": ("NetCashProvidedByUsedInOperatingActivities",
                       "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"),
    "temettu": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends",
                "PaymentsOfDividendsMinorityInterest"),
    "geri_alim": ("PaymentsForRepurchaseOfCommonStock",),
    "hisse_ihraci": ("ProceedsFromIssuanceOfCommonStock",
                     "ProceedsFromIssuanceOfCommonStockNetOfIssuanceCosts"),
    # --- pay ---
    "seyreltilmis_adet": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    # `dei:EntityCommonStockSharesOutstanding` BİLEREK YOK: onun `end`i KAPAK tarihidir,
    # mali dönem sonu değil. Zincire konunca dönem-başına doldurma onu boş bir "mali yıl"
    # satırı olarak çerçeveye sokuyordu (ölçüldü: AAPL 20 → 37 satır, 17'si TAMAMEN NaN)
    # ve `son_iki_yil` "cari yıl" diye o boş satırı döndürüyordu. Bir zincirin halkaları
    # aynı TARİH ANLAMINI taşımalıdır.
    "hisse_adedi": ("CommonStockSharesOutstanding",),
}

# Anlık (stok) kalemler: `start` taşımazlar, dönem uzunluğu sınanmaz.
ANLIK = {
    "varliklar", "donen_varliklar", "yukumlulukler", "kisa_vadeli_yukumluluk", "ozkaynak",
    "nakit", "stok", "alacak", "borc_satici", "uzun_vadeli_borc", "dagitilmamis_kar",
    "serefiye", "duran_varlik", "kiralama_yukumlulugu", "emeklilik_yukumlulugu",
    "emeklilik_varligi", "hisse_adedi",
}

_ETIKET_TERS = {etiket: ad for ad, zincir in CONCEPTS.items() for etiket in zincir}


def _bos() -> pd.DataFrame:
    d = pd.DataFrame(columns=["kalem", "etiket", "end", "start", "val", "filed", "form", "fy", "fp"])
    d.attrs["kaynak"] = "bos"
    return d


def distill(raw: dict) -> pd.DataFrame:
    """Ham `companyfacts` JSON'unu `CONCEPTS`in seçtiği satırlara indir (damıtma).

    Yalnız USD birimli us-gaap kalemleri alınır; `shares` birimli pay adetleri de
    taşınır. Bilinmeyen etiketler DÜŞER — 500 kalemin tamamını taşımak önbelleği
    ham JSON kadar büyütürdü (bkz. modül docstring'i, karar 3).
    """
    facts = (raw or {}).get("facts") or {}
    satirlar: list[dict] = []
    for taksonomi in ("us-gaap",):
        for etiket, blok in (facts.get(taksonomi) or {}).items():
            kalem = _ETIKET_TERS.get(etiket)
            if kalem is None:
                continue
            for birim, kayitlar in (blok.get("units") or {}).items():
                if birim not in ("USD", "shares"):
                    continue
                for k in kayitlar:
                    if k.get("filed") is None or k.get("end") is None or k.get("val") is None:
                        continue
                    try:
                        val = float(k["val"])
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if not math.isfinite(val):
                        continue
                    satirlar.append({
                        "kalem": kalem, "etiket": etiket, "end": k["end"],
                        "start": k.get("start"), "val": val,
                        "filed": k["filed"], "form": k.get("form") or "",
                        "fy": k.get("fy"), "fp": k.get("fp") or "",
                    })
    if not satirlar:
        return _bos()
    d = pd.DataFrame(satirlar)
    for kol in ("end", "start", "filed"):
        d[kol] = pd.to_datetime(d[kol], errors="coerce")
    d = d.dropna(subset=["end", "filed"]).sort_values(["kalem", "end", "filed"])
    return d.reset_index(drop=True)


def facts(symbol: str, *, refresh: bool = False, max_age_days: int = MAX_AGE_DAYS,
          fetch: bool = True) -> pd.DataFrame:
    """Sembolün damıtılmış XBRL kalemleri (önbellekli, fail-soft).

    `fetch=False` ÖNBELLEK-YALNIZ okur ve ağa HİÇ çıkmaz — pano/uç yolları bunu
    kullanmalıdır (BULGU-59-8: okuma yolu ağ çağrısı ödememeli). Önbellek yoksa
    boş çerçeve döner; "veri yok" ile "ölçmedim" ayrımı `attrs["kaynak"]`tadır.
    """
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]{0,14}", symbol):
        return _bos()
    dest = CACHE_DIR / f"{symbol.upper()}.csv"
    taze = False
    if dest.exists():
        yas = (pd.Timestamp.now("UTC").tz_localize(None)
               - pd.Timestamp(dest.stat().st_mtime, unit="s")).days
        taze = yas < max_age_days

    if not fetch or (not refresh and taze):
        if not dest.exists():
            out = _bos()
            out.attrs["kaynak"] = "onbellek-yok"
            return out
        return _oku(dest)

    cik = ticker_to_cik(symbol)
    if cik is None:
        logger.warning(f"[edgar] {symbol} için CIK bulunamadı")
        return _oku(dest) if dest.exists() else _bos()
    raw = _edgar_get(FACTS_URL.format(cik=cik), timeout=60.0)
    if not isinstance(raw, dict):
        logger.warning(f"[edgar] {symbol} companyfacts indirilemedi — bayat kopya kullanılıyor")
        return _oku(dest) if dest.exists() else _bos()
    d = distill(raw)
    if d.empty:
        return _oku(dest) if dest.exists() else _bos()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(dest, d.to_csv(index=False))
    except OSError as e:                                  # pragma: no cover - disk hatası
        logger.warning(f"[edgar] {symbol} önbelleğe yazılamadı: {e}")
    d.attrs["kaynak"] = "ag"
    return d


def warm(symbols: list[str], *, refresh: bool = False,
         as_of: str | None = None, workers: int = 4) -> dict | None:
    """Kuyrukla toplu ısıtma; ağ trafiği ortak `insider._get` sınırından geçer.

    Yalnız damıtılmış CSV ve küçük tarihli medyan JSON'u kalır. Bir şirketin
    başarısızlığı diğerlerini durdurmaz; önbellek ve ağ kaynağı ayrı sayılır.
    Yinelenemeyen sembol listesi, geçersiz `as_of`/`workers` → None (Tur-78, R3-B7).
    """
    from nihai.ai.fundamentals import universe_medians

    try:
        cutoff = pd.Timestamp(as_of or pd.Timestamp.now("UTC").date()).date().isoformat()
        syms = sorted({s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()})
        workers = max(1, min(int(workers), 8))
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None

    def one(sym: str) -> tuple[str, str, list[dict]]:
        try:
            d = facts(sym, refresh=refresh)
            annual = annual_frame(d, as_of=cutoff)
            rows = [{"_donem_sonu": str(t.date()), **row.to_dict()}
                    for t, row in annual.iterrows()]
            return sym, str(d.attrs.get("kaynak", "bos")), rows
        except Exception as exc:
            logger.warning(f"[edgar] {sym} toplu ısıtma başarısız: {exc}")
            return sym, "hata", []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, syms))
    history = {sym: rows for sym, _, rows in results if rows}
    median = universe_medians(history)
    snapshot = {"as_of": cutoff, "evren": syms, "n_sembol": len(history),
                "kapsam": "evren_yillik_vekil_sektor_ve_dusuk_BM_filtresi_yok",
                **median}
    if history:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(CACHE_DIR / f"_medians_{cutoff}.json",
                          json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
    return {"istenen": len(syms), "olculen": len(history),
            "eksik": [s for s in syms if s not in history],
            "kaynaklar": {s: source for s, source, _ in results}, "medyan": snapshot}


def median_snapshot(*, as_of: str | None = None) -> dict | None:
    """Önbellekteki en yeni GEÇMİŞ medyan; ileri tarihli özet asla okunmaz."""
    try:
        cutoff = pd.Timestamp(as_of or pd.Timestamp.now("UTC").date()).date().isoformat()
        for path in sorted(CACHE_DIR.glob("_medians_????-??-??.json"), reverse=True):
            if path.stem.removeprefix("_medians_") > cutoff:
                continue
            result = json.loads(path.read_text("utf-8"))
            if isinstance(result, dict) and result.get("as_of", "9999") <= cutoff:
                return result
    except (OSError, ValueError, TypeError):
        pass
    return None


def _oku(dest: Path) -> pd.DataFrame:
    try:
        d = pd.read_csv(dest, parse_dates=["end", "start", "filed"])
    except (OSError, ValueError) as e:                    # pragma: no cover
        logger.warning(f"[edgar] {dest.name} okunamadı: {e}")
        return _bos()
    d.attrs["kaynak"] = "onbellek"
    return d


def bilinen(d: pd.DataFrame, *, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """`as_of` gününde BİLİNEN satırlar (`filed <= as_of`). PIT kapısı burasıdır."""
    if d.empty:
        return d
    sinir = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now("UTC").tz_localize(None)
    if sinir.tzinfo is not None:
        sinir = sinir.tz_localize(None)
    return d[pd.DatetimeIndex(d["filed"]) <= sinir]


def annual_frame(symbol_or_facts: str | pd.DataFrame, *,
                 as_of: str | pd.Timestamp | None = None,
                 fetch: bool = True) -> pd.DataFrame:
    """Yıllık (10-K) kalemler: satır = mali yıl sonu, kolon = kanonik kalem.

    Seçim kuralları — üçü de sessiz kalırsa yanlış sayı üretirdi:
    * yalnız **10-K** formu (10-Q'nun yıllık kolonları aynı `end`e ikinci bir değer yazar),
    * dönemsel kalemlerde uzunluk **300-400 gün** (10-K içindeki çeyrek satırları eler),
    * aynı `end` için **`as_of`ta bilinen EN SON beyan** (yeniden beyan hakkı saklı).

    `attrs["etiket"]` hangi kalemin hangi us-gaap etiketinden geldiğini taşır.
    """
    d = (symbol_or_facts if isinstance(symbol_or_facts, pd.DataFrame)
         else facts(str(symbol_or_facts), fetch=fetch))
    d = bilinen(d, as_of=as_of)
    if d.empty:
        out = pd.DataFrame()
        out.attrs["etiket"] = {}
        return out
    d = d[d["form"].astype(str).str.startswith("10-K")]
    if d.empty:
        out = pd.DataFrame()
        out.attrs["etiket"] = {}
        return out

    kolonlar: dict[str, pd.Series] = {}
    etiketler: dict[str, dict[str, int]] = {}
    for kalem, zincir in CONCEPTS.items():
        alt = d[d["kalem"] == kalem]
        if alt.empty:
            continue
        if kalem not in ANLIK:
            sure = (pd.DatetimeIndex(alt["end"]) - pd.DatetimeIndex(alt["start"])).days
            alt = alt[(sure >= YIL_GUN_MIN) & (sure <= YIL_GUN_MAX)]
            if alt.empty:
                continue
        # ZİNCİR **DÖNEM BAŞINA** UYGULANIR, kalem başına DEĞİL. Ölçüldü (AAPL):
        # `PaymentsOfDividendsCommonStock` zincirde önde ama 2017'de BİTER;
        # `PaymentsOfDividends` 2026'ya kadar gider. Tek etiket seçen kural (zincir
        # sırası ya da "en çok satır") 2018 sonrası TÜM temettüleri NaN yapıyordu —
        # sayı üretmemek de yanlış sayı üretmenin bir biçimidir. Taksonomi zaten
        # dönem dönem yenilenir (ASC 606: `SalesRevenueNet` → `RevenueFromContract...`).
        seri = pd.Series(dtype=float)
        sayac: dict[str, int] = {}
        for etiket in zincir:
            p = alt[alt["etiket"] == etiket]
            if p.empty:
                continue
            # aynı dönem sonu için en son BEYAN (filed) kazanır — yeniden beyan hakkı
            p = p.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
            s = p.set_index(pd.DatetimeIndex(p["end"]))["val"]
            yeni = s[~s.index.isin(seri.index)]
            if yeni.empty:
                continue
            sayac[etiket] = int(len(yeni))
            seri = pd.concat([seri, yeni])
        if seri.empty:
            continue
        kolonlar[kalem] = seri.sort_index()
        etiketler[kalem] = sayac

    if not kolonlar:
        out = pd.DataFrame()
        out.attrs["etiket"] = {}
        out.attrs["karisik"] = []
        return out
    out = pd.DataFrame(kolonlar).sort_index()
    out.attrs["etiket"] = etiketler
    # Birden çok etiketten beslenen kalemler GİZLENMEZ: tanım kayması buradadır.
    out.attrs["karisik"] = sorted(k for k, v in etiketler.items() if len(v) > 1)
    return out


def son_iki_yil(symbol_or_facts: str | pd.DataFrame, *,
                as_of: str | pd.Timestamp | None = None,
                fetch: bool = True,
                annual: pd.DataFrame | None = None) -> tuple[dict, dict]:
    """(cari_yil, onceki_yil) sözlükleri — Piotroski/Beneish gibi ÇİFT dönem isteyenler için.

    Dönem sonları BİRBİRİNİ İZLEMEK ZORUNDA DEĞİLDİR (bir yıl 10-K gecikirse boşluk
    olur); bu yüzden her iki sözlük `_donem_sonu` taşır ve çağıran farkı görebilir.

    Tur-81 (Codex triyajı, madde B): `annual` verilirse yıllık çerçeve YENİDEN KURULMAZ.
    Pano `/api/quant/fundamentals` ucu `annual_frame`i çağırıp hemen ardından bu
    fonksiyonu çağırıyordu → AYNI ham kayıttan AYNI `as_of` ile AYNI tablo iki kez
    süzülüp sıralanıyordu (Codex ölçümü: AAPL'nin 5.981 ham satırında ikinci kuruluş
    **~147 ms/istek**). Varsayılan `None` = eski davranış; çağıranın geçtiği çerçeve
    aynı girdilerden üretilmiş olmalıdır (uç tam olarak öyle yapar)."""
    d = annual_frame(symbol_or_facts, as_of=as_of, fetch=fetch) if annual is None else annual
    if d.empty:
        return {}, {}
    satirlar = [{"_donem_sonu": str(idx.date()),
                 **{k: (None if pd.isna(v) else float(v)) for k, v in satir.items()}}
                for idx, satir in d.iterrows()]
    cari = satirlar[-1]
    onceki = satirlar[-2] if len(satirlar) >= 2 else {}
    return cari, onceki


def filings(symbol: str, *, forms: tuple[str, ...] | None = None,
            limit: int = 500) -> pd.DataFrame:
    """Sembolün son dosyalamaları: [form, filed, items, accession, rapor_tarihi].

    `items` kolonu 8-K'nın MADDE KODLARINI taşır ("4.02", "5.02,9.01" gibi) — listenin
    180 ve 181. maddeleri tam olarak budur ve serbest metin ayrıştırmaya GEREK YOKTUR.
    Ölçüldü (2026-09-07, AAPL): 103 adet 8-K, `items` dolu; ayrıca ihraççının kendi
    `submissions` bloğunda **SC 13G/SCHEDULE 13G** satırları da görünüyor (madde 182).
    SC 13D (aktivist) bu blokta ÇIKMAYABİLİR — çıkmazsa yokluğu "aktivist yok" diye
    OKUNMAMALIDIR; `attrs["formlar"]` gerçekte ne görüldüğünü taşır.
    """
    cik = ticker_to_cik(symbol)
    if cik is None:
        return pd.DataFrame(columns=["form", "filed", "items", "accession", "rapor_tarihi"])
    blok = _edgar_get(SUBMISSIONS_URL.format(cik=cik))
    rec = ((blok or {}).get("filings") or {}).get("recent") or {}
    if not rec.get("form"):
        return pd.DataFrame(columns=["form", "filed", "items", "accession", "rapor_tarihi"])
    d = pd.DataFrame({
        "form": rec.get("form", []),
        "filed": pd.to_datetime(rec.get("filingDate", []), errors="coerce"),
        "items": rec.get("items", [""] * len(rec["form"])),
        "accession": rec.get("accessionNumber", []),
        "rapor_tarihi": pd.to_datetime(rec.get("reportDate", []), errors="coerce"),
    }).dropna(subset=["filed"])
    gorulen = sorted(set(d["form"].astype(str)))
    if forms:
        d = d[d["form"].astype(str).str.startswith(tuple(forms))]
    out = d.sort_values("filed", ascending=False).head(limit).reset_index(drop=True)
    out.attrs["formlar"] = gorulen
    return out


def item_filings(d: pd.DataFrame, item: str) -> pd.DataFrame:
    """8-K madde koduna göre süz (`"4.02"` → yeniden beyan, `"5.02"` → yönetici ayrılışı).

    Kod eşleşmesi TAM PARÇA üzerindedir: `"4.02"` araması `"4.021"`i getirmez ve
    `"1.04"`ün içindeki `"4.0"`ya takılmaz.
    """
    if d.empty or "items" not in d.columns:
        return d
    kod = item.strip()
    maske = d["items"].fillna("").astype(str).apply(
        lambda s: kod in [p.strip() for p in s.split(",")])
    return d[maske]


def _cli() -> None:  # pragma: no cover - elle çalıştırılan araç
    import argparse

    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()          # Tur-24: pipe'a yazınca cp1254'e düşüp ÇÖKMESİN
    p = argparse.ArgumentParser(description="SEC EDGAR companyfacts köprüsü")
    p.add_argument("symbols", nargs="*")
    p.add_argument("--warm", action="store_true", help="sembolleri, yoksa S&P 500 evrenini ısıt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--as-of", default=None)
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    if a.warm:
        syms = [s.strip() for token in a.symbols for s in token.split(",") if s.strip()]
        if not syms:
            from nihai.data.pit_universe import sp500_as_of
            from nihai.execution.screen import sp500_tickers
            syms = sp500_as_of(a.as_of) if a.as_of else sp500_tickers()
        print(json.dumps(warm(syms, refresh=a.refresh, as_of=a.as_of, workers=a.workers),
                         ensure_ascii=False, allow_nan=False, indent=2))
        return
    for sym in (a.symbols or ["AAPL"]):
        d = annual_frame(facts(sym, refresh=a.refresh), as_of=a.as_of)
        if d.empty:
            print(f"{sym}: veri yok")
            continue
        print(f"\n=== {sym} === {d.shape[0]} mali yıl, {d.shape[1]} kalem")
        print(f"  son dönem: {d.index[-1].date()}")
        print("  etiketler:", json.dumps(d.attrs.get("etiket", {}), ensure_ascii=False)[:300])
        print(d.tail(3).to_string())


if __name__ == "__main__":  # pragma: no cover
    _cli()
