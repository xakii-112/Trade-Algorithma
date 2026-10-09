"""quant_cekirdek.py — Tur-101: quant çekirdeğinin KENDİ tasarım kararları — çıkış, boyut, portföy.

Tur-97 `quant_trend`i (nihai/signals/quant.py) canlı karar yoluna koydu; Tur-98..100 yalnız σ'yı (ML) sınadı. Çekirdeğin
geri kalan kararları — pozisyondan NASIL çıkılır, skorun büyüklüğü boyuta girer mi, sinyaller TEK portföyde nasıl
birleşir — hiç sınanmadı. Bu modül onları ön-kayıtla sınar.

TEŞHİS (koşudan önce; kod, teori ve 2026-09-27 canlı anlık görüntüsü — hiçbir kolun getirisine bakılmadan):
  D1 ÇIKIŞ UFKU. Pozisyon yalnız 3σ İZLEYEN stopla çıkar (tepe − 3σ·fiyat; tepe gün-içi yüksekten). Sürüklenmesiz Brown
     hareketinde tepeden a kadar geri çekilmenin beklenen süresi (a/σ)² (Taylor 1975) → 3σ için ≈ 9 işlem günü; sinyalin
     ufku 21/63/252 gün. Stop sonrası skor hâlâ > 0 ise ertesi karar barında yeniden girilir → trend sürerken
     "sat–geri al" döngüsü ve her döngüde 2 × %0,1 maliyet.
  D2 SKOR BOYUTU. Hurst-Ooi-Pedersen (2017) pozisyonu ufuk pozisyonlarının ORTALAMASIDIR (skor ∈ {±⅓, ±1}); çekirdek
     yalnız işaretini kullanır: ⅓ ile 1 aynı riski alır. 2026-09-27: odak listesinin 23 LONG'unun 11'i ⅓.
  D3 FX İŞARET KONVANSİYONU. Uzun-yalnız kararı hisse sürüklenmesine dayanır (Tur-97); döviz çiftinde "uzun" kotasyon
     yönüdür (EURUSD uzun = USDEUR kısa) → yalnız simetrik kural kotasyondan bağımsızdır.
  D4 CANLI PORTFÖY KAPASİTESİ. Canlı süreç 2026-09-27 11:13'te `quant_trend` koduyla başladı; ilk seans 2026-09-28.
     Odak 30 sembolün 23'ü LONG ve tam riskte (%1,5 / 3σ) toplam 5,84× özkaynak nominal ister; tavan net yön 2×
     (Reg T brüt 2×), sektör başı 3 (TECH'te 13 aday). Canlı liste 520 sembol (odak 30 + geniş 490, iki iş parçacığı
     katmanı, ortak portföy): ~2× nominal dolana dek HANGİ sembole önce ulaşılırsa o alınır — bileşimi iş parçacığı
     zamanlaması belirler; geniş katman quant için hiç ölçülmedi.
  D5 ÖLÇÜM ↔ CANLI. Tur-97'nin ABD_izleme ölçümü 30 BAĞIMSIZ hesabın (her biri tam risk) ortalamasıdır; canlı ise tek
     portföy ve paylaşılan tavanlardır. `backtest_portfolio` tavanları uygular ama sembolleri sözlük sırasıyla gezer.

ÖN-KAYIT (2026-09-27; aşağıdaki kolların hiçbirinin getirisi görülmeden sabitlendi)
 BÖLÜM I — sembol başı motor (`backtest_symbol` 1d, Tur-97 düzeneği: kill-switch ve AI kapısı kapalı, %0,1/taraf,
 kapanış→kapanış MTM, piyasa = eşit ağırlıklı enstrüman ortalaması), 14 piyasa + ABD_izleme, ÜÇ pencere:
   A arsiv_2001_08 : 2001-01-01 → 2009-01-01 — YENİ; bu depoda quant için HİÇ ölçülmedi. Hazırlık 1998'den, Tur-96
                     temizliği aynen (bayat bar, tek-bar aralık onarımı, kripto dışı |günlük log getiri| > 0,5 → dışla).
   B oos_2009_17   : Tur-97 ÖD penceresi birebir.   C test_2018_26 : Tur-97 TEST penceresi birebir.
   Enstrüman penceredeyse: A ve B'de değerlendirme ≥ 500 bar (Tur-97 kuralı); motor ısınması ≥ 60 bar.
 Kollar (yalnız araştırma yamaları; üretim kodu bu ölçüm için DEĞİŞMEZ):
   q0   canlı çekirdek. ÖN-KOŞUL: B ve C'de Tur-97 sonuç dosyalarıyla günlük getiri farkı 0 (tutmazsa HÜKÜM YOK).
   q1   SİNYAL ÇIKIŞI: izleyen stop kapalı; başlangıç 3σ stopu SABİT kalır (risk birimi ve felaket koruması); karar
        kapanışında skor ≤ 0 (LONG) ise ertesi açılışta çık (QUANT_SINYAL).
   q0s  SKOR BOYUTU: q0 + giriş riski × |skor|.
   q2   q1 + skor boyutu (paket).
   BİLGİ (kapısız): q1n = sinyal çıkışı, stop YOK (boyut birimi yine 3σ; stopun kendi payını ayırır);
        Forex simetrik q0k / q1k (QUANT_KISA yalnız Forex enstrümanlarında).
 Hipotezler: H1 = q1 − q0 · H2 = q0s − q0 · H12 = q2 − q0 (yalnız H1 VE H2 geçerse kapılı, aksi halde bilgi).
 Birincil ölçü: yıllık Sharpe farkı, piyasa z'si Memmel (2003). Yan: MDD, CAGR, işlem/yıl, tutma süresi, yeniden giriş.
 KAPI G (her kapılı H, ÜÇ pencerenin ÜÇÜNDE): (i) korelasyon düzeltmeli Stouffer birleşik z > 1,645;
   (ii) ΔSharpe > 0 olan piyasa ≥ ⌈⅔·n⌉ (n = o pencerede ölçülen piyasa); (iii) ABD_izleme ΔSharpe > 0.
 Bağlantı: G'yi geçen kol quant çekirdeğinde VARSAYILAN olur (eski davranış bayrakla korunur); geçmeyen kol yalnız
   bu dosyada kalır.
 BÖLÜM II — portföy (canlı paritesi): ABD_izleme sembolleri TEK portföyde, canlı tavanlarla (Reg T, net yön 2×, sektör 3,
   max_open 50), `backtest_portfolio` döngüsünün birebir kopyası (plan kapalı + dizin ısınmasıyla backtest_portfolio'ya
   PARİTE 0; tutmazsa hüküm yok). Pencereler A/B/C (sembol o pencerede veri taşıyorsa).
   p0       canlı: tam risk, `.env` liste sırası.   p0_perm: 8 rastgele sıra (tohum 1..8) — canlı sıra belirsizliği.
   p1       QUANT PORTFÖY PLANI: her karar gününde LONG adaylar skor azalan (eşitlikte liste sırası) sıralanır; sektör (3)
            ve sayı (50) tavanları bu sırayla doldurulur; yeni girişlerin risk çarpanı
            m = min(1, (C − N_açık) / Σ n_i), n_i = tam riskli nominal/özkaynak = risk%·(düşüş freni)/(3σ_i),
            C = net yön tavanı; planda olmayan sembole giriş yok.
   KAPI P (q0 çekirdeğiyle, üç pencerenin üçünde): Sharpe(p1) ≥ medyan Sharpe(p0_perm) VE MDD(p1) ≤ medyan MDD(p0_perm).
   P geçerse plan canlıya bağlanır — bileşimi zamanlamadan kurtarır; kenar iddiası DEĞİLDİR. Bilgi: Memmel z (p1 − p0),
   aynı ölçüm q1/q0s/q2 çekirdekleriyle.
 SADAKAT (kapısız; ölçüm gerektirmez): quant pozisyonu yalnız ODAK katmanında açılır — quant'ın ölçüldüğü evren odak
   listesidir; geniş katman (oto-tarayıcı çıktısı, 4 saatte bir değişir) quant için ölçülmedi.

EK ÖN-KAYIT — Z DOĞRULAMA PENCERESİ (2026-09-27, A/B/C sonuçları GÖRÜLDÜKTEN SONRA, Z'nin hiçbir çıktısı görülmeden):
 A/B/C'de kapı P geçti; ayrıca kapısız bilgi kolunda sinyal çıkışlı çekirdek (q1) portföyde liste sırasıyla q0'ın çok üstünde
 çıktı (S 0,38/1,08/0,93 ↔ −0,19/0,47/0,60) ama planla q0-planı geçmedi (0,05/0,79/0,80 ↔ 0,05/0,88/0,75), sembol başı H1 ise
 C'de kaldı. Bu gözlem SONRADANDIR (Tur-23) → hüküm yalnız dokunulmamış veride verilir:
   Z dogrulama_1990_00: 1990-01-01 → 2001-01-01, hazırlık 1987'den, ABD_izleme sembollerinden o pencerede veri taşıyanlar,
     aynı temizlik ve aynı pencere kuralı (≥ 500 değerlendirme barı).
   Z1 (P'nin doğrulaması): q0 planı Z'de de kapı P'yi geçmeli; GEÇMEZSE plan canlıya BAĞLANMAZ (A/B/C kararı geri alınır).
   Z2 (portföy çekirdeği): q1_plan − q0_plan Memmel z > 1,645 → sinyal çıkışı için YENİ ön-kayıtlı ileri sınama önerilir;
     bu tur canlı çekirdeği her durumda DEĞİŞTİRMEZ (sembol başı H1 kaldı).
   Z3 (bilgi): q1 liste sırası ↔ q1'in 8 rastgele sırası (liste sırası şanslı bir sıra mı?) — A/B/C'de de koşulur.
   Yan ölçü: plan çarpanı m'nin dağılımı (ortalama, en küçük, m < 1 günlerin payı).

CLI:
  python -m research.quant_cekirdek --asama motor --isci 8     # Bölüm I: enstrüman × pencere × kol (+ işlem defteri)
  python -m research.quant_cekirdek --asama portfoy --isci 6   # Bölüm II: portföy kolları
  python -m research.quant_cekirdek --asama rapor              # data/quant_cekirdek_karnesi.json + özet
"""
from __future__ import annotations

import contextlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research import coklu_piyasa as C
from research import quant_karar as QKR
from research import quant_kayma as QK

ROOT = Path(__file__).resolve().parents[1]
SONUC_DIR = C.SONUC_DIR / "quant101"
KARNE = ROOT / "data" / "quant_cekirdek_karnesi.json"
ARSIV_BAS = "1998-01-01"
PENCERELER: dict[str, tuple[str, str | None]] = {
    "A": ("2001-01-01", "2009-01-01"), "B": ("2009-01-01", "2018-01-01"), "C": ("2018-01-01", None)}
DOGRULAMA: dict[str, tuple[str, str]] = {"Z": ("1990-01-01", "2001-01-01")}     # yalnız Bölüm II (ek ön-kayıt)
HAZIRLIK_BAS = {"A": ARSIV_BAS, "Z": "1987-01-01"}
PORTFOY_PENCERE = ("A", "B", "C", "Z")
MIN_DEGER_BAR = 500                         # A/B değerlendirme uzunluğu (Tur-97 OOS_MIN_BAR)
KOLLAR = ("q0", "q1", "q0s", "q2", "q1n")
FX_KOLLAR = ("q0k", "q1k")
HIPOTEZLER: dict[str, tuple[str, str, bool | None]] = {   # ad: (aday, taban, kapılı; None = H1∧H2'ye bağlı)
    "H1": ("q1", "q0", True), "H2": ("q0s", "q0", True), "H12": ("q2", "q0", None),
    "q1n": ("q1n", "q0", False), "q2_q1": ("q2", "q1", False), "fx_q0k": ("q0k", "q0", False),
    "fx_q1k": ("q1k", "q1", False)}
KAPI = {"birlesik_z": 1.645, "pay": 2.0 / 3.0, "parite": 0.0}
PERM_TOHUM = tuple(range(1, 9))
PORTFOY_CEKIRDEK = ("q0", "q1", "q0s", "q2")
SINYAL_CIKISI = {"q1", "q2", "q1n", "q1k"}
SKOR_BOYUTU = {"q0s", "q2"}


# ---------------------------------------------------------------------------------------------------
# Veri — pencere başına (df, t0). B ve C Tur-97 ile BİREBİR (parite ön-koşulu); A yeni hazırlık.
# ---------------------------------------------------------------------------------------------------
def _pencere(pen: str) -> tuple[str, str | None]:
    return PENCERELER[pen] if pen in PENCERELER else DOGRULAMA[pen]


def hazirla_arsiv(sym: str, pen: str = "A") -> tuple[pd.DataFrame | None, str | None]:
    """Arşiv penceresi (A: 1998 → 2009, Z: 1987 → 2001) hazırlığı: `C.hazirla` ile aynı sıra (bayat bar → aralık
    onarımı → ölçek) ve `C.kusur_nedeni` ile aynı kusur kuralı. Dönen: (df, None) ya da (None, dışlama nedeni)."""
    from research import piyasalar as P
    ham = P.yukle(sym)
    w = ham[(ham.index >= HAZIRLIK_BAS[pen]) & (ham.index < pd.Timestamp(_pencere(pen)[1]))]
    if len(w) < C.SICAK + 300:
        return None, f"yetersiz geçmiş ({len(w)} bar)"
    kripto = sym in P.PIYASALAR["Kripto"]["semboller"]
    if not kripto:
        r = np.abs(np.diff(np.log(w["close"].to_numpy(float))))
        if r.size and float(r.max()) > C.KUSUR_ESIGI:
            i = int(np.argmax(r))
            return None, f"|log getiri| {r[i]:.2f} ({w.index[i + 1].date()}) > {C.KUSUR_ESIGI}"
    w = w[~P.bayat_bar_maskesi(w)]
    if not kripto:
        w, _ = P.aralik_onar(w, C.KUSUR_ESIGI)
    return P.olcekle(w), None


def pencere_verisi(sym: str, pen: str) -> tuple[pd.DataFrame, int] | None:
    """(df, t0): motor df'in tamamını görür, t0 değerlendirmenin ilk karar barıdır."""
    bas, son = _pencere(pen)
    if pen in HAZIRLIK_BAS:
        df, _ = hazirla_arsiv(sym, pen)
        if df is None:
            return None
    else:
        df = C.hazirla(sym)
        if son is not None:
            df = df[df.index < pd.Timestamp(son)]
    t0 = int((df.index < pd.Timestamp(bas)).sum())
    if pen != "C":                        # Tur-97 ÖD kuralı: ısınma ≥ 60, değerlendirme ≥ 500 bar
        t0 = max(t0, 60)
        if len(df) - t0 < MIN_DEGER_BAR:
            return None
    elif len(df) - t0 < 2:                # Tur-97 TEST kuralı: evren (`kusur_nedeni`) dışında şart yok
        return None
    return df, t0


def skor_once(df: pd.DataFrame) -> dict[pd.Timestamp, float]:
    """{bar t+1 zamanı: skor[t]} — çıkış denetimi bar t+1'de, karar t kapanışındaki skorla (nedensel)."""
    from nihai.signals import quant as Q
    skor, _ = Q.seriler(df["close"].to_numpy(float))
    return {pd.Timestamp(df.index[i + 1]): float(skor[i]) for i in range(len(df) - 1)}


# ---------------------------------------------------------------------------------------------------
# Motor yamaları (araştırma). Orijinaller Tur-97'nin ORTAK sözlüğünde süreç başına bir kez (BULGU-96-5).
# ---------------------------------------------------------------------------------------------------
_DURUM: dict = {"skor_once": {}, "son_skor": None}


@contextlib.contextmanager
def cekirdek(kol: str, skorlar: dict[str, dict[pd.Timestamp, float]]):
    """`kol` çekirdeğini motora giydirir. skorlar: {sembol: skor_once(df)} (yalnız sinyal çıkışı okur)."""
    import nihai.execution.backtest as B
    from nihai.config import CONFIG
    from nihai.risk.position_manager import PositionManager as PM
    from nihai.risk.risk_manager import RiskManager as RM
    from nihai.signals.strategy import is_quant_label
    o_cik = QK._ORIJINAL.setdefault("pm_check_exit", PM.check_exit)
    o_izl = QK._ORIJINAL.setdefault("pm_trail", PM._apply_trailing)
    o_boy = QK._ORIJINAL.setdefault("rm_size", RM.size_stock)
    o_gs = QK._ORIJINAL.setdefault("gs", B.generate_signal)
    sinyal, boyut, stopsuz = kol in SINYAL_CIKISI, kol in SKOR_BOYUTU, kol == "q1n"
    kisa_once = CONFIG.strategy.quant_allow_short

    def check_exit(self, t, *, bar_high, bar_low, bar_open, bar_close=None, now_dt):
        p = t.pos
        kw = {"bar_high": bar_high, "bar_low": bar_low, "bar_open": bar_open, "bar_close": bar_close,
              "now_dt": now_dt}
        if not is_quant_label(p.strategy):
            return o_cik(self, t, **kw)
        s = _DURUM["skor_once"].get(p.symbol, {}).get(pd.Timestamp(now_dt))
        if s is not None and math.isfinite(s) and (s <= 0 if p.side == "LONG" else s >= 0):
            return ("QUANT_SINYAL", bar_open)          # karar t kapanışında, infaz t+1 açılışında
        if stopsuz:
            eski, p.stop = p.stop, 0.0
            try:
                return o_cik(self, t, **kw)
            finally:
                p.stop = eski
        return o_cik(self, t, **kw)

    def apply_trailing(self, t):
        if is_quant_label(t.pos.strategy):
            return                                      # q1: izleyen stop yok, başlangıç stopu sabit
        return o_izl(self, t)

    def generate_signal(*a, **kw):
        sig = o_gs(*a, **kw)
        _DURUM["son_skor"] = (sig.metadata.get("quant") or {}).get("skor")
        return sig

    def size_stock(self, entry, stop, sector="UNKNOWN", *, confidence_multiplier=1.0, **kw):
        s = _DURUM["son_skor"]
        if s is not None and math.isfinite(s):
            confidence_multiplier = confidence_multiplier * abs(float(s))
        return o_boy(self, entry, stop, sector, confidence_multiplier=confidence_multiplier, **kw)

    yamalar: list[tuple[object, str, object, object]] = []
    if sinyal:
        yamalar += [(PM, "check_exit", check_exit, o_cik), (PM, "_apply_trailing", apply_trailing, o_izl)]
    if boyut:
        yamalar += [(B, "generate_signal", generate_signal, o_gs), (RM, "size_stock", size_stock, o_boy)]
    _DURUM.update(skor_once=skorlar if sinyal else {}, son_skor=None)
    for hedef, ad, f, _ in yamalar:
        setattr(hedef, ad, f)
    object.__setattr__(CONFIG.strategy, "quant_allow_short", kol in FX_KOLLAR)
    try:
        yield
    finally:
        for hedef, ad, _, orj in yamalar:
            setattr(hedef, ad, orj)
        object.__setattr__(CONFIG.strategy, "quant_allow_short", kisa_once)
        _DURUM.update(skor_once={}, son_skor=None)


# ---------------------------------------------------------------------------------------------------
# BÖLÜM I — enstrüman başı motor
# ---------------------------------------------------------------------------------------------------
def _dosya(piyasa: str, sym: str) -> Path:
    return SONUC_DIR / QK.dosya(piyasa, sym, "k101_").name


def defter_ozeti(isl: list[dict], n_bar: int) -> dict:
    """İşlem defteri → D1 ölçüleri: tutma (bar), stop sonrası yeniden giriş aralığı, çıkış nedenleri, giriş nominali."""
    tut = [x["b"] - x["a"] for x in isl if x["b"] is not None]
    ara = [isl[k + 1]["a"] - isl[k]["b"] for k in range(len(isl) - 1) if isl[k]["b"] is not None]
    neden: dict[str, int] = {}
    for x in isl:
        neden[x["neden"] or "ACIK"] = neden.get(x["neden"] or "ACIK", 0) + 1
    return {"islem": len(isl), "yil": n_bar / 252.0, "tutma": tut, "ara": ara, "neden": neden,
            "nominal": [x["u"] * x["F"] / x["E"] for x in isl if x["E"] > 0]}


def enstruman_kos(arg: tuple[str, str]) -> str:
    piyasa, sym = arg
    C._isci_hazirla()
    t = time.time()
    out: dict = {"piyasa": piyasa, "sembol": sym, "kos": {}, "parite": {}}
    eski = QK._oku(piyasa, sym)
    kollar = KOLLAR + (FX_KOLLAR if piyasa == "Forex" else ())
    for pen in PENCERELER:
        v = pencere_verisi(sym, pen)
        if v is None:
            continue
        df, t0 = v
        sk = {sym: skor_once(df)}
        for kol in kollar:
            with cekirdek(kol, sk), QKR.defter() as kay:
                r = C._algoritma_kos(sym, df, t0, "quant_trend")
            s = pd.Series(r["r"].astype(float), index=pd.DatetimeIndex(r["tarih"]))
            out["kos"][f"{kol}_{pen}"] = {"s": s, "islem": r["islem"],
                                           "defter": defter_ozeti(QKR.islemler(kay, df.index), len(df) - t0)}
        onceki = (eski or {}).get("kos", {}).get({"B": "quant_trend_oos", "C": "quant_trend"}.get(pen, ""))
        if onceki is not None:
            q0 = out["kos"][f"q0_{pen}"]["s"]
            a = onceki["s"].reindex(q0.index).to_numpy(float)
            out["parite"][pen] = (float(np.nanmax(np.abs(q0.to_numpy() - a))) if len(q0) == len(onceki["s"])
                                  and np.isfinite(a).all() else math.inf)
    SONUC_DIR.mkdir(parents=True, exist_ok=True)
    with open(_dosya(piyasa, sym), "wb") as f:
        pickle.dump(out, f)
    return f"{piyasa}/{sym}: {len(out['kos'])} koşu, parite {out['parite']}, {round(time.time() - t, 1)} sn"


# ---------------------------------------------------------------------------------------------------
# BÖLÜM II — TEK portföy (canlı tavanları). `backtest_portfolio` döngüsünün birebir kopyası + plan kancası.
# ---------------------------------------------------------------------------------------------------
def plan_kur(adaylar: list[tuple[str, float, float, str]], acik: list, sira: dict[str, int], E: float,
             risk_pct: float, tavan: float, sektor_tavani: int, max_acik: int,
             skor_boyutu: bool = False) -> tuple[list[str], float]:
    """Quant portföy planı. adaylar: [(sembol, skor, σ, sektör)] yalnız pozisyonu OLMAYAN LONG adaylar.
    Dönen: (seçilenler sıralı, yeni girişlerin risk çarpanı m ∈ [0, 1])."""
    sektor: dict[str, int] = {}
    for p in acik:
        sektor[p.sector] = sektor.get(p.sector, 0) + 1
    secilen: list[tuple[str, float, float]] = []
    for s, sk, sg, sek in sorted(adaylar, key=lambda x: (-x[1], sira.get(x[0], 10 ** 6))):
        if len(acik) + len(secilen) >= max_acik or sektor.get(sek, 0) >= sektor_tavani:
            continue
        sektor[sek] = sektor.get(sek, 0) + 1
        secilen.append((s, sk, sg))
    talep = sum(risk_pct * (abs(sk) if skor_boyutu else 1.0) / (3.0 * sg) for _, sk, sg in secilen)
    net0 = sum(p.signed_notional for p in acik) / E if E > 0 else 0.0
    m = 1.0 if talep <= 0 else min(1.0, max(tavan - net0, 0.0) / talep)
    return [s for s, _, _ in secilen], m


def portfoy_kos(data: dict[str, pd.DataFrame], sira: list[str], *, bas: str | None = None, plan: bool = False,
                kol: str = "q0", warmup_bars: int = 60) -> dict:
    """Paylaşılan portföy. bas=None → `backtest_portfolio` ile aynı DİZİN ısınması (parite); bas verilirse o
    tarihten önce giriş yok ve özkaynak noktası yazılmaz. Dönen: özkaynak serisi + maruziyet/defter ölçüleri."""
    from datetime import date, datetime

    from nihai.config import CONFIG, INITIAL_EQUITY, get_sector
    from nihai.execution import backtest as B
    from nihai.risk.position_manager import PositionManager
    from nihai.risk.risk_manager import PortfolioState, RiskManager
    from nihai.signals import quant as Q
    from nihai.signals.strategy import _SharedIndicators, quant_capala
    data = {s: data[s] for s in sira if s in data and data[s] is not None and not data[s].empty}
    portfolio = PortfolioState(equity=INITIAL_EQUITY, ath_equity=INITIAL_EQUITY, cash_balance=INITIAL_EQUITY)
    portfolio.last_session_date = date.min             # backtest_portfolio ile aynı (BULGU-101-5)
    risk_mgr = RiskManager(portfolio)
    pmgr = PositionManager(portfolio, B._bt_slippage(), record_logs=False)
    prep: dict = {}
    for s, d in data.items():
        dup = d.index.duplicated(keep=False)
        pos_map = {ts_: j for j, (ts_, du) in enumerate(zip(d.index, dup, strict=True)) if not du}
        adv_a = (B._causal_adv(d["volume"].to_numpy(dtype=float), CONFIG.backtest.impact_adv_bars)
                 if "volume" in d.columns else None)
        skor, sig = Q.seriler(d["close"].to_numpy(float))
        prep[s] = (pos_map, d["open"].to_numpy(float), d["high"].to_numpy(float), d["low"].to_numpy(float),
                   d["close"].to_numpy(float), d.index.to_pydatetime(), _SharedIndicators(d), adv_a, skor, sig)
    konum = {s: k for k, s in enumerate(sira)}
    caps = CONFIG.portfolio
    tarih: list = []
    eq: list[float] = []
    maruz: list[float] = []
    n_acik: list[int] = []
    tutulan: dict[str, int] = {}
    m_gun: list[float] = []
    islem = 0
    all_ts = sorted(set().union(*[set(d.index) for d in data.values()]))
    t_bas = pd.Timestamp(bas) if bas is not None else None
    for ts in all_ts:
        if t_bas is not None and ts < t_bas:
            continue
        gecerli = {}
        for s in data:
            i = prep[s][0].get(ts)
            if i is None or i < warmup_bars or i >= len(data[s]) - 1:
                continue
            gecerli[s] = i
        sirali = list(gecerli)
        secim: set[str] | None = None
        m = 1.0
        if plan:
            acik_s = {p.symbol for p in portfolio.open_positions}
            ad = [(s, float(prep[s][8][i]), float(prep[s][9][i]), get_sector(s)) for s, i in gecerli.items()
                  if s not in acik_s and math.isfinite(prep[s][8][i]) and prep[s][8][i] > 0
                  and math.isfinite(prep[s][9][i]) and prep[s][9][i] > 0]
            risk_pct = risk_mgr._current_risk_pct(CONFIG.risk.per_trade_risk_pct_directional)
            secilen, m = plan_kur(ad, list(portfolio.open_positions), konum, portfolio.equity, risk_pct,
                                  caps.max_net_directional_pct, caps.max_per_sector_positions,
                                  caps.max_open_positions, skor_boyutu=kol in SKOR_BOYUTU)
            secim = set(secilen)
            sirali = secilen + [s for s in gecerli if s not in secim]
            if secilen:
                m_gun.append(m)
        for s in sirali:
            i = gecerli[s]
            pos_map, o_a, h_a, l_a, c_a, ts_list, shared_full, adv_a, _, _ = prep[s]
            bars = {s: (float(o_a[i + 1]), float(h_a[i + 1]), float(l_a[i + 1]), float(c_a[i + 1]))}
            liquidity = ({s: {"adv_shares": float(adv_a[i]) if adv_a is not None else None,
                              "bar_sigma": B._bar_sigma(shared_full, i, float(c_a[i]))}}
                         if pmgr.slip.impact_coef else None)
            portfolio.roll_session_if_new_day(ts_list[i + 1].date())
            closed_now = pmgr.update_all_with_bars(bars, now=ts_list[i + 1], liquidity_by_symbol=liquidity)
            if any(p.symbol == s for p in portfolio.open_positions):
                continue
            if closed_now and CONFIG.backtest.reentry_policy != "same_bar":
                continue
            if secim is not None and s not in secim:
                continue
            sg = B.generate_signal(s, data[s].iloc[: i + 1], strategy_mode="quant_trend",
                                   shared=shared_full.upto(i + 1))
            if sg.is_actionable():
                sg = quant_capala(sg, float(o_a[i + 1]))
                rc = risk_mgr.size_stock(entry=float(o_a[i + 1]), stop=sg.stop, sector=get_sector(s), symbol=s,
                                         side=sg.action.value, confidence_multiplier=m)
                if rc.approved:
                    signal_price = sg.price
                    sg.price = float(o_a[i + 1])
                    pmgr.open_from_signal(sg, rc.sizing, now=ts_list[i], asset_type="stock", sector=get_sector(s),
                                          signal_price=signal_price,
                                          strategy_name=f"{sg.metadata.get('strategy', 'EventBreakout')} V1",
                                          adv_shares=(float(adv_a[i]) if adv_a is not None else None),
                                          bar_sigma=B._bar_sigma(shared_full, i, float(o_a[i + 1])))
                    islem += 1
        E = B._mtm_equity(portfolio)
        tarih.append(ts)
        eq.append(E)
        maruz.append(sum(abs(p.signed_notional) for p in portfolio.open_positions) / E if E > 0 else 0.0)
        n_acik.append(len(portfolio.open_positions))
        for p in portfolio.open_positions:
            tutulan[p.symbol] = tutulan.get(p.symbol, 0) + 1
    for s, d in data.items():
        for pos in [p for p in list(portfolio.open_positions) if p.symbol == s]:
            pmgr.force_close(pos, float(d["close"].iloc[-1]), "BT_END",
                             d.index[-1].to_pydatetime() if hasattr(d.index[-1], "to_pydatetime") else datetime.now())
    ey = np.asarray([INITIAL_EQUITY] + eq, float)
    toplam = sum(tutulan.values()) or 1
    return {"r": pd.Series(ey[1:] / ey[:-1] - 1.0, index=pd.DatetimeIndex(tarih)), "eq": eq, "islem": islem,
            "maruziyet": float(np.mean(maruz)) if maruz else 0.0, "ort_acik": float(np.mean(n_acik)) if n_acik else 0.0,
            "farkli_sembol": len(tutulan), "hhi": float(sum((v / toplam) ** 2 for v in tutulan.values())),
            "gun": len(tarih),
            "m": ({"ort": float(np.mean(m_gun)), "min": float(np.min(m_gun)),
                   "kucuk_pay": float(np.mean(np.asarray(m_gun) < 1.0)), "gun": len(m_gun)} if m_gun else None)}


def izleme_verisi(pen: str, semboller: list[str]) -> dict[str, pd.DataFrame]:
    """Pencerenin portföy verisi: Bölüm I'in pencere kuralıyla (değerlendirmede ≥ 500 bar ya da C'de ≥ 250)."""
    out = {}
    for s in semboller:
        v = pencere_verisi(s, pen)
        if v is not None:
            out[s] = v[0]
    return out


def portfoy_gorevleri() -> list[tuple[str, str, str, int]]:
    """(pencere, çekirdek, düzen, tohum). düzen: liste | perm | plan. Tohum yalnız perm'de anlamlı."""
    g: list[tuple[str, str, str, int]] = []
    for pen in PORTFOY_PENCERE:
        for kol in PORTFOY_CEKIRDEK:
            g.append((pen, kol, "liste", 0))
            g.append((pen, kol, "plan", 0))
        g += [(pen, kol, "perm", t) for kol in ("q0", "q1") for t in PERM_TOHUM]
    return g


def portfoy_kos_gorev(arg: tuple[str, str, str, int]) -> str:
    pen, kol, duzen, tohum = arg
    C._isci_hazirla()
    t = time.time()
    liste = [s for pz, s in QK.evren()[0] if pz == QK.IZLEME]
    data = izleme_verisi(pen, liste)
    sira = [s for s in liste if s in data]
    if duzen == "perm":
        sira = [sira[k] for k in np.random.default_rng(tohum).permutation(len(sira))]
    sk = {s: skor_once(d) for s, d in data.items()}
    with cekirdek(kol, sk):
        r = portfoy_kos(data, sira, bas=_pencere(pen)[0], plan=duzen == "plan", kol=kol)
    r["sembol"] = sira
    SONUC_DIR.mkdir(parents=True, exist_ok=True)
    with open(SONUC_DIR / f"pf_{pen}_{kol}_{duzen}_{tohum}.pkl", "wb") as f:
        pickle.dump(r, f)
    return f"portföy {pen}/{kol}/{duzen}/{tohum}: {len(sira)} sembol, {r['islem']} işlem, {round(time.time() - t, 1)} sn"


def portfoy_paritesi(pen: str = "C", n: int = 8) -> dict:
    """Plan kapalı + dizin ısınması → `backtest_portfolio` ile özkaynak eğrisi farkı (ön-koşul)."""
    from nihai.execution.backtest import backtest_portfolio
    C._isci_hazirla()
    liste = [s for pz, s in QK.evren()[0] if pz == QK.IZLEME][:n]
    data = izleme_verisi(pen, liste)
    sira = [s for s in liste if s in data]
    r = backtest_portfolio(sira, "1d", data={s: data[s] for s in sira}, strategy_mode="quant_trend",
                           disable_kill_switch=True, warmup_bars=60)
    k = portfoy_kos(data, sira, bas=None, plan=False)
    bizim = np.asarray(k["eq"], float)
    onlar = np.asarray(r.equity_curve[1: 1 + bizim.size], float)
    return {"n_sembol": len(sira), "gun": int(bizim.size), "fark": float(np.max(np.abs(bizim - onlar))),
            "islem_motor": int(r.trades_count), "islem_kopya": int(k["islem"])}


# ---------------------------------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------------------------------
def _piyasa_blogu(pz: str, semboller: list[str]) -> tuple[dict, dict, dict]:
    """→ ({kol_pen: EW seri}, {kol_pen: birleşik defter ölçüleri}, {pen: parite max})."""
    seri: dict[str, list[pd.Series]] = {}
    df_: dict[str, dict] = {}
    parite: dict[str, float] = {}
    for s in semboller:
        p = _dosya(pz, s)
        if not p.exists():
            continue
        with open(p, "rb") as f:
            q = pickle.load(f)
        for pen, v in q["parite"].items():
            parite[pen] = max(parite.get(pen, 0.0), v)
        for ad, k in q["kos"].items():
            seri.setdefault(ad, []).append(k["s"])
            d = df_.setdefault(ad, {"islem": 0, "yil": 0.0, "tutma": [], "ara": [], "neden": {}, "nominal": [],
                                    "enstruman": 0})
            dd = k["defter"]
            d["islem"] += dd["islem"]
            d["yil"] += dd["yil"]
            d["tutma"] += dd["tutma"]
            d["ara"] += dd["ara"]
            d["nominal"] += dd["nominal"]
            d["enstruman"] += 1
            for n, c in dd["neden"].items():
                d["neden"][n] = d["neden"].get(n, 0) + c
    return {ad: C._ew(v) for ad, v in seri.items()}, df_, parite


def defter_metrik(d: dict) -> dict:
    tut = np.asarray(d["tutma"], float)
    ara = np.asarray(d["ara"], float)
    tn = sum(d["neden"].values()) or 1
    return {"islem_yil_enstruman": round(d["islem"] / max(d["yil"], 1e-9), 2),
            "tutma_medyan": float(np.median(tut)) if tut.size else None,
            "tutma_ort": round(float(tut.mean()), 1) if tut.size else None,
            "yeniden_giris_2bar": round(float((ara <= 2).mean()), 3) if ara.size else None,
            "neden_pay": {n: round(c / tn, 3) for n, c in sorted(d["neden"].items())},
            "giris_nominal_ort": round(float(np.mean(d["nominal"])), 3) if d["nominal"] else None}


def bolum1(kull: list[tuple[str, str]]) -> dict:
    piyasalar: dict[str, list[str]] = {}
    for pz, s in kull:
        piyasalar.setdefault(pz, []).append(s)
    out: dict = {"piyasalar": {}, "birlesik": {}, "parite": {}}
    fark: dict = {}
    for pz, semboller in piyasalar.items():
        gun = QK._gun(pz)
        ew, dft, par = _piyasa_blogu(pz, semboller)
        for pen, v in par.items():
            out["parite"][pen] = max(out["parite"].get(pen, 0.0), v)
        blok: dict = {}
        for pen in PENCERELER:
            kol = {ad[: -len(pen) - 1]: s for ad, s in ew.items() if ad.endswith("_" + pen)}
            if "q0" not in kol:
                continue
            b: dict = {"metrik": {k: {**C.seri_metrik(s.to_numpy(), gun), **defter_metrik(dft[f"{k}_{pen}"])}
                                  for k, s in kol.items()}}
            for h, (aday, taban, _) in HIPOTEZLER.items():
                if aday in kol and taban in kol:
                    b[h] = QKR.sharpe_fark(kol[aday], kol[taban], gun)
                    ortak = kol[aday].index.intersection(kol[taban].index)
                    fark.setdefault((h, pen), {})[pz] = kol[aday].loc[ortak] - kol[taban].loc[ortak]
            blok[pen] = b
        out["piyasalar"][pz] = blok
    for h in HIPOTEZLER:
        out["birlesik"][h] = {}
        for pen in PENCERELER:
            bl = {pz: b[pen][h] for pz, b in out["piyasalar"].items() if pen in b and h in b[pen]}
            if not bl:
                continue
            fk = fark.get((h, pen), {})
            bz = QKR.birlesik_z({pz: m["z"] for pz, m in bl.items()}, fk)
            _, S = QKR.korelasyon_toplami({pz: fk[pz] for pz in bl if bl[pz]["se"]})
            out["birlesik"][h][pen] = {**bz, "olumlu_piyasa": sum(1 for m in bl.values() if (m["ds"] or 0) > 0),
                                       "n_piyasa": len(bl), "izleme_ds": (bl.get(QK.IZLEME) or {}).get("ds"),
                                       "mde_sonradan": QKR.mde({pz: m["se"] for pz, m in bl.items()}, S)}
    return out


def g_kapisi(b: dict) -> bool:
    """Tek pencere: birleşik z > 1,645 VE olumlu piyasa ≥ ⌈⅔·n⌉ VE ABD_izleme ΔSharpe > 0."""
    return bool(b.get("z") is not None and b["z"] > KAPI["birlesik_z"]
                and b["olumlu_piyasa"] >= math.ceil(KAPI["pay"] * b["n_piyasa"] - 1e-9)
                and b.get("izleme_ds") is not None and b["izleme_ds"] > 0)


def hukum(b1: dict) -> dict:
    parite_ok = bool(b1["parite"]) and all(v <= KAPI["parite"] for v in b1["parite"].values())
    out: dict = {"parite_gecti": parite_ok}
    for h, (_, _, kapili) in HIPOTEZLER.items():
        kos = {pen: g_kapisi(b) for pen, b in b1["birlesik"].get(h, {}).items()}
        out[h] = {"kapili": kapili, "pencere": kos,
                  "gecti": parite_ok and set(kos) == set(PENCERELER) and all(kos.values())}
    h1, h2 = out["H1"]["gecti"], out["H2"]["gecti"]
    out["H12"]["kapili"] = h1 and h2
    out["aday"] = ("q2" if h1 and h2 and out["H12"]["gecti"] else "q1" if h1 else "q0s" if h2 else None)
    return out


def bolum2() -> dict:
    from nihai.config import CONFIG
    out: dict = {}
    for pen in PORTFOY_PENCERE:
        blok: dict = {}
        oku = {}
        for p in SONUC_DIR.glob(f"pf_{pen}_*.pkl"):
            _, _, kol, duzen, tohum = p.stem.split("_")
            with open(p, "rb") as f:
                oku[(kol, duzen, int(tohum))] = pickle.load(f)
        if not oku:
            continue
        def met(r: dict) -> dict:
            return {**C.seri_metrik(r["r"].to_numpy(), 252), "islem": r["islem"], "maruziyet": round(r["maruziyet"], 3),
                    "ort_acik": round(r["ort_acik"], 2), "farkli_sembol": r["farkli_sembol"],
                    "hhi": round(r["hhi"], 3), "n_sembol": len(r["sembol"]), "m": r.get("m")}
        for kol in PORTFOY_CEKIRDEK:
            for duzen in ("liste", "plan"):
                if (kol, duzen, 0) in oku:
                    blok[f"{kol}_{duzen}"] = met(oku[(kol, duzen, 0)])
            if (kol, "liste", 0) in oku and (kol, "plan", 0) in oku:
                blok[f"{kol}_plan_fark"] = QKR.sharpe_fark(oku[(kol, "plan", 0)]["r"], oku[(kol, "liste", 0)]["r"], 252)
        if ("q1", "plan", 0) in oku and ("q0", "plan", 0) in oku:
            blok["q1plan_q0plan"] = QKR.sharpe_fark(oku[("q1", "plan", 0)]["r"], oku[("q0", "plan", 0)]["r"], 252)
        if ("q1", "liste", 0) in oku and ("q0", "plan", 0) in oku:
            blok["q1liste_q0plan"] = QKR.sharpe_fark(oku[("q1", "liste", 0)]["r"], oku[("q0", "plan", 0)]["r"], 252)
        for kol in ("q0", "q1"):
            perm = [met(oku[k]) for k in sorted(oku) if k[0] == kol and k[1] == "perm"]
            if not perm:
                continue
            sh = [p["sharpe"] for p in perm if p.get("sharpe") is not None]
            md = [p["mdd_pct"] for p in perm if p.get("mdd_pct") is not None]
            blok[f"{kol}_perm"] = {"sharpe": sh, "mdd_pct": md, "sharpe_medyan": float(np.median(sh)),
                                   "mdd_medyan": float(np.median(md)), "sharpe_aralik": [min(sh), max(sh)],
                                   "mdd_aralik": [min(md), max(md)]}
        pl, pm = blok.get("q0_plan"), blok.get("q0_perm")
        if pl and pm:
            blok["kapi_p"] = bool(pl["sharpe"] >= pm["sharpe_medyan"] and pl["mdd_pct"] <= pm["mdd_medyan"])
        out[pen] = blok
    out["tavanlar"] = {"net_yon": CONFIG.portfolio.max_net_directional_pct,
                       "sektor": CONFIG.portfolio.max_per_sector_positions,
                       "max_acik": CONFIG.portfolio.max_open_positions,
                       "risk_islem": CONFIG.risk.per_trade_risk_pct_directional}
    out["kapi_p_gecti"] = all(out.get(pen, {}).get("kapi_p") is True for pen in PENCERELER)
    z = out.get("Z", {})
    out["z1"] = z.get("kapi_p") is True
    out["z2"] = bool((z.get("q1plan_q0plan") or {}).get("z") is not None and z["q1plan_q0plan"]["z"] > KAPI["birlesik_z"])
    out["plan_baglanir"] = out["kapi_p_gecti"] and out["z1"]
    return out


def rapor(kull: list[tuple[str, str]]) -> dict:
    b1 = bolum1(kull)
    karne = {"uretim": time.strftime("%Y-%m-%d %H:%M"), "pencereler": PENCERELER, "kapi": KAPI,
             "hipotezler": {h: {"aday": a, "taban": b, "kapili": k} for h, (a, b, k) in HIPOTEZLER.items()},
             "maliyet_taraf": C.maliyet_orani(), "bolum1": b1, "hukum": hukum(b1), "bolum2": bolum2()}
    return karne


def ozet_md(karne: dict) -> str:
    b1, hk = karne["bolum1"], karne["hukum"]
    sat = [f"parite (Tur-97, B/C): {b1['parite']} → {'GEÇTİ' if hk['parite_gecti'] else 'HÜKÜM YOK'}", ""]
    sat.append("| H | pencere | birleşik z | olumlu/n | izleme ΔS | MDE | kapı |")
    sat.append("|---|---|---|---|---|---|---|")
    for h, pens in b1["birlesik"].items():
        for pen, b in pens.items():
            sat.append(f"| {h} | {pen} | {b.get('z')} | {b['olumlu_piyasa']}/{b['n_piyasa']} | {b.get('izleme_ds')} | "
                       f"{b.get('mde_sonradan')} | {'✓' if g_kapisi(b) else '✗'} |")
    sat.append("")
    sat.append("hüküm: " + ", ".join(f"{h}={'GEÇTİ' if v['gecti'] else 'kaldı'}" for h, v in hk.items()
                                     if isinstance(v, dict)) + f" · aday={hk['aday']}")
    b2 = karne["bolum2"]
    for pen in PORTFOY_PENCERE:
        bl = b2.get(pen)
        if not bl:
            continue
        pm, p1 = bl.get("q0_perm", {}), bl.get("q1_perm", {})
        sat.append(f"portföy {pen}: q0 liste S {bl.get('q0_liste', {}).get('sharpe')} · plan S "
                   f"{bl.get('q0_plan', {}).get('sharpe')} · perm medyan S {pm.get('sharpe_medyan')} "
                   f"(aralık {pm.get('sharpe_aralik')}) · kapı P {bl.get('kapi_p')} | q1 liste S "
                   f"{bl.get('q1_liste', {}).get('sharpe')} · q1 perm {p1.get('sharpe_aralik')} · q1plan−q0plan "
                   f"{bl.get('q1plan_q0plan')}")
    sat.append(f"kapı P (A/B/C): {b2.get('kapi_p_gecti')} · Z1: {b2.get('z1')} · Z2: {b2.get('z2')} · "
               f"plan bağlanır: {b2.get('plan_baglanir')}")
    return "\n".join(sat)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Tur-101 quant çekirdeği — çıkış, skor boyutu, portföy planı")
    ap.add_argument("--asama", required=True, choices=["motor", "portfoy", "parite", "rapor"])
    ap.add_argument("--isci", type=int, default=8)
    ap.add_argument("--sembol", default=None, help="yalnız bu sembol (duman testi)")
    a = ap.parse_args()
    C._isci_hazirla()
    kull, _ = QK.evren()
    if a.sembol:
        kull = [(pz, s) for pz, s in kull if s == a.sembol]
    if a.asama == "motor":
        C._havuz(enstruman_kos, kull, a.isci)
    elif a.asama == "portfoy":
        C._havuz(portfoy_kos_gorev, portfoy_gorevleri(), a.isci)
    elif a.asama == "parite":
        print(json.dumps(portfoy_paritesi(), ensure_ascii=False))
    else:
        karne = rapor(kull)
        karne["parmak_izi"] = C.girdi_parmak_izi(sorted({s for _, s in kull}))
        KARNE.write_bytes(json.dumps(karne, ensure_ascii=False, indent=1, default=str).encode("utf-8"))
        print(ozet_md(karne))
        print(f"karne → {KARNE}")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    _cli()
