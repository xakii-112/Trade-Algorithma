"""
tax_report.py — IRS FORM 8949 DÖKÜMÜ (Tur-75, Madde 254). KARAR-DIŞI.

Kapanan sanal işlemleri (`data/trades.csv`) muhasebeciye verilebilecek Form 8949
sütun düzeninde CSV'ye döker. **Bu bir vergi TAVSİYESİ değil bir BİÇİM DÖNÜŞÜMÜDÜR**;
sistem sanal (kâğıt) işlem üretir, dolayısıyla çıktı gerçek bir beyan değil bir
ŞABLONDUR ve her satır `virtual=True` taşır.

## Üç vergi kuralı KODA yazılıdır (yanlış bilgi üretmemek için)

1. **§1233 — AÇIĞA SATIŞ HER ZAMAN KISA VADELİDİR.** Tutma süresi bir yılı aşsa bile
   short işlemin kazancı/kaybı kısa vadelidir (26 U.S.C. §1233(b)). Tutma süresine
   bakıp short'u uzun vadeli sayan bir döküm, vergiyi EKSİK hesaplatır.
2. **§1256 SÖZLEŞMELERİ FORM 8949'A GİRMEZ.** Nakit uzlaşılı endeks opsiyonları
   (SPX/NDX/VIX/RUT/XSP) Form **6781**'e girer ve 60/40 muamelesi görür. Bu döküm
   onları AYRI bir kovaya alır ve 8949 dosyasına YAZMAZ — Tur-58'in aynı kaynağı
   (`options_analyzer.settlement_style`) okunur, ikinci bir liste tanımlanmaz.
3. **UZUN VADE SINIRI "BİR YILDAN FAZLA"DIR.** Tam 365 gün KISA vadedir; sınır
   `> 365 gün`tür (IRS Pub. 544: "more than one year"). `>=` yazan bir uygulama
   sınırdaki her işlemi yanlış tarafa koyar.

## Lot yöntemi (Madde 6.2)

`position` giriş/çıkış çiftini korur; `fifo` en eski, `hifo` en yüksek birim
maliyetli açık LONG lotu seçer. HIFO, ABD'de belirli tanımlama (specific
identification, Treas. Reg. §1.1012-1(c)(3)) gerektirir: satış anında lotun
belirtilmesi ve brokerın makul sürede yazılı teyidiyle yeterli kayıt gerekir.
Broker varsayılanı genellikle FIFO'dur. Sistem sanal işlem ürettiği için bu
eşleme de yalnız bir ŞABLONDUR. Kaynak: https://www.irs.gov/publications/p550
Kontrat kimliği olmayan opsiyonlar FIFO/HIFO'da da pozisyon eşlemesinde kalır.

## Yapılmayan ve NEDEN yapılmadığı

**Wash-sale düzeltmesi (kolon f/g) HESAPLANMAZ.** `state.wash_sale_check` canlı
karar yolunda 31 günlük karantinayı UYGULAR ama zararın DOLAR TUTARINI bir sonraki
alımın maliyet tabanına TAŞIMAZ; taşımayan bir sistemin dökümüne "W" kodu yazmak,
yapılmamış bir düzeltmeyi yapılmış göstermek olurdu. Satırlar `wash_sale_candidate`
bayrağı taşır (aynı sembolde 30 gün içinde ZARARLA kapanış + yeniden giriş) →
muhasebeci nereye bakacağını bilir, sistem uydurmaz.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR
from nihai.console import ensure_utf8_stdout
from nihai.risk.options_analyzer import settlement_style

TRADES_CSV = Path(DATA_DIR) / "trades.csv"
# IRS Pub. 544: "more than one year" → 365 gün KISA, 366 gün UZUN vadedir.
LONG_TERM_DAYS = 365
WASH_SALE_WINDOW_DAYS = 30

FORM_8949_COLS = [
    "description",        # (a) 100 sh. AAPL
    "date_acquired",      # (b)
    "date_sold",          # (c)
    "proceeds",           # (d)
    "cost_basis",         # (e)
    "adjustment_code",    # (f)
    "adjustment_amount",  # (g)
    "gain_loss",          # (h) = (d) − (e) + (g)
    # 8949 formunun kendisinde OLMAYAN, izlenebilirlik için taşınan alanlar:
    "part", "symbol", "side", "holding_days", "virtual", "wash_sale_candidate",
]


# Stdlib ile pandas'ın BİREBİR aynı sonucu verdiği ISO biçimi. `fromisoformat` pandas'ın REDDETTİĞİ
# hafta tarihini ("2026-W01-1"), 'X' gibi ayırıcıyı ve saniyeli ofseti KABUL eder (Codex incelemesi,
# bağımsız doğrulandı: eskiden dışlanan işlem rapora giriyordu) → onlar hızlı yola hiç girmez.
_ISO_TS = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)?(?:Z|[+-]\d{2}:\d{2})?")


def _parse_ts(v: object) -> datetime | None:
    # Tur-80: satır başına `pd.to_datetime` biçim TAHMİNİ yapıyordu (724 işlemde 1448 çağrı ≈ 0,9 sn,
    # `build`in %60'ı). Yukarıdaki biçimde stdlib aynı naive-UTC değeri verir; kalan her girdi eski
    # yola düşer → davranış birebir.
    if isinstance(v, str) and _ISO_TS.fullmatch(v):
        try:
            dt = datetime.fromisoformat(v)
        except ValueError:
            dt = None
        if dt is not None:
            return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo is not None else dt
    try:
        t = pd.to_datetime(v, errors="coerce", utc=True)
    except (TypeError, ValueError):
        return None
    if t is pd.NaT or pd.isna(t):
        return None
    return t.to_pydatetime().replace(tzinfo=None)


def _wash_adaylari(df: pd.DataFrame) -> set[int]:
    """ZARARLA kapanan bir işlemin sembolünde ±30 gün içinde YENİ giriş var mı?

    Aday demek düzeltme demek DEĞİLDİR (modül docstring'i). Yalnız muhasebecinin
    bakacağı satırları işaretler.
    """
    adaylar: set[int] = set()
    for _sym, grup in df.groupby("symbol"):
        girisler = grup["_entry"].dropna().tolist()
        # Tur-80: `iterrows` satır başına karışık tipli Series kuruyordu; sütun zip'i aynı
        # Timestamp/float değerlerini verir (gerçek + sentetik veride aday kümesi özdeş).
        for idx, pnl, cikis, giris in zip(grup.index, grup["_pnl"], grup["_exit"],
                                          grup["_entry"], strict=True):
            if not (pnl < 0) or pd.isna(cikis):
                continue
            for g in girisler:
                if g == giris:
                    continue
                if abs((g - cikis).days) <= WASH_SALE_WINDOW_DAYS:
                    adaylar.add(int(idx))
                    break
    return adaylar


def _kaynak_satirlari(df: pd.DataFrame, wash: set[int]) -> list[dict]:
    """Satır sözlükleri + wash bayrağı. Tur-80: `itertuples` sözlükleri `iterrows`unkiyle tip ve
    değer olarak birebir (8× hızlı); test bu yardımcıyı iterrows referansıyla değiştirip kıyaslar."""
    kolonlar = list(df.columns)
    return [{**dict(zip(kolonlar, t, strict=True)), "_wash_candidate": int(idx) in wash}
            for idx, t in zip(df.index, df.itertuples(index=False, name=None), strict=True)]


def _lot_eslestir(df: pd.DataFrame, lot_method: str, wash: set[int]) -> tuple:
    """Olayları eşle; girdi değişmez, eksik edinimin maliyeti uydurulmaz.

    Örtüşme sayısı seçilen yöntemin açık LONG lotlarında, sembol/varlık türü
    başına ölçülür. Opsiyonlar kontrat kimliği olmadığından bu ölçüme girmez.
    `position` ölçümü kendi lotunu kapatır; eski çıktı sırası ve değerleri korunur.
    """
    kaynak = _kaynak_satirlari(df, wash)
    sonuc = []
    olaylar = []
    edinimler: dict[str, list[tuple]] = {}
    tutarsiz: list[dict] = []
    fallback = 0
    for idx, r in enumerate(kaynak):
        option = str(r.get("asset_type", "stock")).lower() == "option"
        short = str(r["side"]).strip().upper() in ("SHORT", "SELL", "-1")
        if lot_method == "position" or option or short:
            sonuc.append(r)
        if option or short:
            fallback += int(option and lot_method != "position")
            continue
        if not pd.isna(r["_entry"]) and not pd.isna(r["_exit"]) and r["_exit"] < r["_entry"]:
            # Pozisyon kipi bu satırı zaten atlar (`cikis < giris`). Olay akışına sokulsaydı çıkışı
            # girişinden ÖNCE başka lotları tüketir, girişi de hiç kapanmayan HAYALET lot olup
            # sonraki satışlara maliyet verirdi. Sessizce de atlanmaz: `unmatched`e gerekçesiyle girer.
            try:
                q_ham = abs(float(r["quantity"]))
            except (TypeError, ValueError):
                q_ham = math.nan
            tutarsiz.append({"symbol": str(r["symbol"]), "time": r["_exit"].isoformat(),
                             "quantity": q_ham if math.isfinite(q_ham) else None,
                             "reason": "CIKIS_GIRISTEN_ONCE"})
            continue
        try:
            qty = abs(Decimal(str(r["quantity"])))
        except (InvalidOperation, TypeError, ValueError):
            continue
        if not qty.is_finite() or qty <= 0:
            continue
        # Her olay kendi tarihi/fiyatıyla doğrulanır: eksik çıkış edinimi silmez;
        # eksik/geç edinim de daha önceki satışa hayalî maliyet sağlayamaz.
        for oncelik, tarih_kol, fiyat_kol in ((0, "_entry", "entry_price"), (1, "_exit", "exit_price")):
            try:
                deger = float(r[fiyat_kol])
            except (TypeError, ValueError):
                continue
            if pd.isna(r[tarih_kol]) or not math.isfinite(deger):
                continue
            olaylar.append((r[tarih_kol], oncelik, idx, qty, deger))
            if oncelik == 0:
                edinimler.setdefault(str(r["symbol"]), []).append((idx, r[tarih_kol]))

    acik: dict[tuple[str, str], dict[int, dict]] = {}
    overlap = 0
    unmatched = list(tutarsiz) if lot_method != "position" else []
    onceki_zaman = None
    satis_anindaki_sayilar: dict[tuple[str, str], int] = {}
    # Eşit zamanda bütün edinimler önce; kalan eşitlikte CSV sırası belirleyicidir.
    for zaman, oncelik, idx, qty, fiyat in sorted(olaylar):
        if zaman != onceki_zaman:
            satis_anindaki_sayilar.clear()
            onceki_zaman = zaman
        r = kaynak[idx]
        sym = str(r["symbol"])
        anahtar = (sym, str(r.get("asset_type", "stock")).lower())
        lotlar = acik.setdefault(anahtar, {})
        if oncelik == 0:
            lotlar[idx] = {"quantity": qty, "entry_price": fiyat, "_entry": zaman}
            continue
        # Eşzamanlı satışlar aynı edinim-sonrası fotoğrafı ölçer; sayım CSV'deki
        # satış sırasına bağlı değildir. Tüketim ise aşağıda sırayla yapılır.
        overlap += int(satis_anindaki_sayilar.setdefault(anahtar, len(lotlar)) > 1)
        if lot_method == "position":
            lotlar.pop(idx, None)
            continue
        # ponytail: satış başına açık lotlar sıralanır; binlerce eşzamanlı lotta heap kullanılabilir.
        sira = sorted(lotlar, key=lambda i: (
            -lotlar[i]["entry_price"] if lot_method == "hifo" else 0,
            lotlar[i]["_entry"], i,
        ))
        for lot_id in sira:
            lot = lotlar[lot_id]
            miktar = min(qty, lot["quantity"])
            aday = fiyat < lot["entry_price"] and any(
                diger_id != lot_id
                and abs((giris.date() - zaman.date()).days) <= WASH_SALE_WINDOW_DAYS
                for diger_id, giris in edinimler.get(sym, [])
            )
            sonuc.append({**r, **lot, "quantity": miktar, "exit_price": fiyat,
                          "_wash_candidate": aday})
            qty -= miktar
            lot["quantity"] -= miktar
            if lot["quantity"] == 0:
                del lotlar[lot_id]
            if qty == 0:
                break
        if qty > 0:
            unmatched.append({"symbol": sym, "time": zaman.isoformat(), "quantity": float(qty)})
    return sonuc, overlap, unmatched, fallback


def build(trades_csv: Path | str | None = None, lot_method: str = "position") -> dict:
    """`trades.csv` → Form 8949 satırları + §1256 kovası + toplamlar.

    Dosya yoksa/boşsa BOŞ sonuç döner (istisna YOK) ve `reason` taşır.
    Geçersiz lot yöntemi veri hatası değil API ihlalidir → ValueError.
    """
    if lot_method not in ("position", "fifo", "hifo"):
        raise ValueError(f"Geçersiz lot_method: {lot_method!r}")
    lot_bilgisi = {"lot_method": lot_method, "n_overlap_dispositions": 0,
                   "unmatched": [], "n_option_position_fallback": 0,
                   "option_position_fallback_reason": None}
    p = Path(trades_csv) if trades_csv is not None else TRADES_CSV
    if not p.exists():
        return {**lot_bilgisi, "rows": [], "section_1256": [], "n": 0, "reason": "DOSYA_YOK"}
    try:
        df = pd.read_csv(p)
    except (OSError, ValueError, pd.errors.ParserError) as e:
        return {**lot_bilgisi, "rows": [], "section_1256": [], "n": 0,
                "reason": f"OKUNAMADI:{type(e).__name__}: {e}"}
    if df.empty:
        return {**lot_bilgisi, "rows": [], "section_1256": [], "n": 0, "reason": "BOS"}

    gerekli = {"symbol", "side", "entry_time", "exit_time", "entry_price",
               "exit_price", "quantity"}
    eksik = gerekli - set(df.columns)
    if eksik:
        return {**lot_bilgisi, "rows": [], "section_1256": [], "n": 0,
                "reason": f"KOLON_EKSIK:{sorted(eksik)}"}

    df = df.copy()
    df["_entry"] = df["entry_time"].map(_parse_ts)
    df["_exit"] = df["exit_time"].map(_parse_ts)
    df["_pnl"] = pd.to_numeric(df.get("realized_pnl", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    wash = _wash_adaylari(df)
    islemler, overlap, unmatched, fallback = _lot_eslestir(df, lot_method, wash)
    lot_bilgisi.update(n_overlap_dispositions=overlap, unmatched=unmatched,
                      n_option_position_fallback=fallback,
                      option_position_fallback_reason="KONTRAT_KIMLIGI_YOK" if fallback else None)

    satirlar: list[dict] = []
    s1256: list[dict] = []
    for r in islemler:
        giris, cikis = r["_entry"], r["_exit"]
        if pd.isna(giris) or pd.isna(cikis) or cikis < giris:
            continue
        try:
            qty = abs(float(r["quantity"]))
            ep = float(r["entry_price"])
            xp = float(r["exit_price"])
        except (TypeError, ValueError):
            continue
        if qty <= 0 or not all(math.isfinite(v) for v in (qty, ep, xp)):
            continue
        option = str(r.get("asset_type", "stock")).lower() == "option"
        multiplier = 100 if option else 1
        short = str(r["side"]).strip().upper() in ("SHORT", "SELL", "-1")
        gun = (cikis - giris).days
        # §1233: açığa satış HER ZAMAN kısa vadeli (tutma süresine BAKILMAZ).
        uzun = (not short) and gun > LONG_TERM_DAYS
        # Short'ta satış ÖNCE olur: hasılat girişten, maliyet çıkıştan.
        hasilat = (ep if short else xp) * qty * multiplier
        maliyet = (xp if short else ep) * qty * multiplier
        kayit = {
            "description": f"{qty:g} {'sh.' if str(r.get('asset_type', 'stock')).lower() == 'stock' else 'ct.'} {r['symbol']}",
            "date_acquired": giris.date().isoformat(),
            "date_sold": cikis.date().isoformat(),
            "proceeds": round(hasilat, 2),
            "cost_basis": round(maliyet, 2),
            "adjustment_code": "",
            "adjustment_amount": 0.0,
            "gain_loss": round(hasilat - maliyet, 2),
            "part": "II (uzun vade)" if uzun else "I (kısa vade)",
            "symbol": str(r["symbol"]),
            "side": "SHORT" if short else "LONG",
            "holding_days": int(gun),
            "virtual": True,
            "wash_sale_candidate": r["_wash_candidate"],
        }
        if option and settlement_style(str(r["symbol"])).get("tax_1256"):
            kayit["part"] = "FORM 6781 (§1256 · 60/40) — 8949'A GİRMEZ"
            s1256.append(kayit)
        else:
            satirlar.append(kayit)

    def _top(rs: list[dict], part_onek: str | None = None) -> float:
        return round(sum(x["gain_loss"] for x in rs
                         if part_onek is None or x["part"].startswith(part_onek)), 2)

    return {
        **lot_bilgisi,
        "rows": satirlar,
        "section_1256": s1256,
        "n": len(satirlar),
        "n_1256": len(s1256),
        "n_wash_candidates": sum(1 for x in satirlar if x["wash_sale_candidate"]),
        "short_term_total": _top(satirlar, "I "),
        "long_term_total": _top(satirlar, "II"),
        "total": _top(satirlar),
        "total_1256": _top(s1256),
        "reason": None,
    }


def write_csv(result: dict, out_path: Path | str) -> Path | None:
    """Form 8949 satırlarını CSV'ye yaz. §1256 satırları AYRI dosyaya (`*_6781.csv`)."""
    out = Path(out_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FORM_8949_COLS)
            w.writeheader()
            w.writerows(result["rows"])
        if result.get("section_1256"):
            alt = out.with_name(out.stem + "_6781" + out.suffix)
            with open(alt, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FORM_8949_COLS)
                w.writeheader()
                w.writerows(result["section_1256"])
    except OSError as e:
        logger.warning(f"[vergi] yazılamadı: {e}")
        return None
    return out


def _cli() -> None:                              # pragma: no cover - operatör aracı
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="IRS Form 8949 dökümü (sanal işlemler)")
    ap.add_argument("--trades", default=str(TRADES_CSV))
    ap.add_argument("--out", default=str(Path(DATA_DIR) / "form_8949.csv"))
    ap.add_argument("--year", type=int, default=None, help="yalnız bu yılın kapanışları")
    ap.add_argument("--lot-method", choices=("position", "fifo", "hifo"), default="position")
    a = ap.parse_args()
    r = build(a.trades, lot_method=a.lot_method)
    if r["reason"]:
        print(f"Döküm üretilemedi: {r['reason']}")
        return
    if a.year is not None:
        r["rows"] = [x for x in r["rows"] if x["date_sold"].startswith(str(a.year))]
        r["section_1256"] = [x for x in r["section_1256"]
                             if x["date_sold"].startswith(str(a.year))]
        r["n"] = len(r["rows"])
    yol = write_csv(r, a.out)
    print(f"Form 8949: {r['n']} satır → {yol}")
    print(f"  Lot yöntemi        : {r['lot_method']}")
    print(f"  Örtüşen satış      : {r['n_overlap_dispositions']}")
    print(f"  Kısa vade (Part I) : {r['short_term_total']:+,.2f} $")
    print(f"  Uzun vade (Part II): {r['long_term_total']:+,.2f} $")
    print(f"  TOPLAM             : {r['total']:+,.2f} $")
    if r["section_1256"]:
        print(f"  §1256 (Form 6781)  : {len(r['section_1256'])} satır, "
              f"{r['total_1256']:+,.2f} $ — 8949'a GİRMEZ")
    if r["n_wash_candidates"]:
        print(f"  Wash-sale ADAYI    : {r['n_wash_candidates']} satır "
              f"(düzeltme HESAPLANMADI — modül docstring'ine bak)")
    print("  UYARI: işlemler SANALDIR; bu bir beyan değil bir şablondur.")


if __name__ == "__main__":                       # pragma: no cover
    _cli()
