"""
orderflow.py — EMİR AKIŞI: footprint · hacim profili · büyük işlem baloncukları · likidite ısı
haritası (Tur-103). KARAR-DIŞI.

DeepCharts/Bookmap tarzı dört görünümün veri tarafı. Hepsi İŞLEM (tick) düzeyindedir: hacim
GERÇEKLEŞTİĞİ fiyata yazılır. `nihai/ai/volume_profile` aynı soruyu bar OHLCV'sinden YAKLAŞIK
kurar (hacmi barın aralığına yayar); buradaki profil yaklaşım değil sayımdır.

Kaynaklar (çağrılarak ölçüldü, 2026-09-28):

| Piyasa | İşlem | Agresör tarafı | Emir defteri (ısı haritası) |
|---|---|---|---|
| ABD hisse `sip` | Alpaca SIP tarihsel, 15 dk gecikmeli, TÜM piyasa (AAPL 30 dk ≈ 27 bin işlem, 2,3 sn) | borsa bildirmez → tick kuralı (TAHMİN) | YOK (ücretsiz L2 yok) |
| ABD hisse `iex` | Alpaca IEX, canlı, konsolide hacmin ~%2-4'ü (AAPL 30 dk ≈ 1 bin işlem) | tick kuralı (TAHMİN) | YOK (IEX yalnız en iyi kotasyon) |
| Kripto (…USDT) | Binance aggTrades, anahtarsız | GERÇEK (`m` = alıcı yapıcı) | Binance depth, 1000 kademe (BTC ≈ ±150-300 $) |

Isı haritası geçmişe dönük ÇEKİLEMEZ: Binance depth yalnız ANLIK görüntüdür. `CryptoRecorder`
sembol ilk istendiğinde başlar, 3 sn'de bir defter görüntüsü alır, son 45 dk'yı tutar ve 10 dk
istek gelmezse durur → ısı haritası sekme açıldığı andan itibaren birikir. İşlemler başlangıçta
son 60 dk için GERİYE doğru doldurulur (en yeni önce; sayfa tavanında en eskiler düşer).

Tick kuralının doğruluğu ÖLÇÜLDÜ (2026-09-28, Tur-103 raporu §3): gerçek agresör tarafının bilindiği
Binance'te son 60 dk BTC %97,5 · ETH %97,2 · SOL %83,6 (sıfır-tik payı %29 / %35 / %59); SIP'te 5 dk'lık
AAPL/SPY/NVDA'da kotasyon tabanlı Lee-Ready ile uyum %84,3 / %79,0 / %81,4. Hisse için gerçek taraf
bilinmediğinden bu bir UYUM ölçüsüdür, doğruluk değil.

ANAYASA: sinyale/eşiğe/META_GATE'e/execution'a bağlanmaz; `signals`/`risk`/`execution` bu modülü
import etmez (test). Ölçüm yokluğu ölçüm gibi sunulmaz: hissede ısı haritası `None` + neden; tick
kuralının sınıflayamadığı (ilk fiyat değişiminden önceki) hacim ayrı sayılır (`siniflanamayan`).
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from loguru import logger

from nihai.ai.option_flow import chain_side, dealer_gamma_exposure
from nihai.ai.volume_profile import VALUE_AREA_PCT, _genislet

BINANCE_URL = "https://api.binance.com"
_HTTP = requests.Session()
_TIMEOUT = 10

BAR_SN = (15, 30, 60, 120, 300, 900)
EN_COK_DAKIKA = 240
EN_COK_ISLEM = 300_000          # tek hisse çekiminin tavanı (en YENİLER kalır: sort=desc)
SIP_GECIKME = timedelta(minutes=16)   # ücretsiz planda son 15 dk SIP kapalı (+1 dk pay; alpaca_history ile aynı)
DIAGONAL_ORAN = 3.0             # footprint dengesizliği: alış[P] ≥ 3 × satış[P−1] (ve simetriği)

# CTA/UTP işlem koşulları: fiyatı o anki açık artırmayı temsil ETMEYEN baskılar — ortalama fiyat (B/W),
# türev fiyatlı (4), QCT (7), düzeltilmiş kapanış (9), nakit/ertesi gün/satıcı (C/N/R), toplu/fiyat
# sapmalı (G/H), resmi açılış/kapanış RAPORU (Q/M: hacmi İKİ KEZ sayar), önceki referans (P), sıra dışı
# bildirim (U/Z), koşullu (V). Footprint hacmi gerçekleştiği FİYATA yazar; bunlar yanlış kademeye yazar.
_HARIC_KOSUL = frozenset("BW479CGHMNPQRUVZ")
_KRIPTO_KOT = ("USDT", "USDC", "FDUSD")


class BinanceSinir(RuntimeError):
    """HTTP 429/418 — Binance ağırlık sınırı; `bekle` sn beklenmeli (418 = IP geçici yasaklı)."""

    def __init__(self, bekle: float, kod: int) -> None:
        super().__init__(f"Binance sınırı (HTTP {kod}), {bekle:.0f} sn bekleniyor")
        self.bekle = bekle


def _binance_get(yol: str, params: dict[str, Any]) -> Any:
    r = _HTTP.get(BINANCE_URL + yol, params=params, timeout=_TIMEOUT)
    if r.status_code in (418, 429):
        try:
            bekle = float(r.headers.get("Retry-After", "60"))
        except ValueError:
            bekle = 60.0
        raise BinanceSinir(bekle, r.status_code)
    r.raise_for_status()
    return r.json()


def _simdi_ms() -> int:
    return int(time.time() * 1000)


def kripto_sembolu(symbol: str) -> str | None:
    """'BTCUSDT' / 'BTC-USD' → Binance sembolü; hisse → None."""
    s = symbol.strip().upper()
    if s.endswith(("-USD", "-USDT")):
        taban = s.split("-", 1)[0]
        return taban + "USDT" if taban.isalnum() else None
    if s.isalnum() and any(s.endswith(k) and len(s) > len(k) for k in _KRIPTO_KOT):
        return s
    return None


# ---------------------------------------------------------------------------
# Saf hesaplar
# ---------------------------------------------------------------------------
def tick_kurali(price: np.ndarray) -> np.ndarray:
    """Tick kuralı: yukarı tik alış (+1), aşağı tik satış (−1), sıfır tik bir öncekinin tarafını taşır;
    ilk fiyat değişiminden ÖNCEKİ işlemler 0 (SINIFLANAMADI — uydurulmaz).

    Lee-Ready'nin kotasyonsuz yarısıdır. ABD borsaları agresör tarafını bildirmez; bu bir TAHMİNDİR
    (doğruluğu Tur-103 raporunda gerçek tarafa karşı ölçüldü)."""
    p = np.asarray(price, dtype=float)
    if p.size == 0:
        return np.zeros(0, dtype=np.int8)
    d = np.sign(np.diff(p, prepend=p[0]))
    son_degisim = np.maximum.accumulate(np.where(d != 0, np.arange(p.size), -1))
    return np.where(son_degisim >= 0, d[np.maximum(son_degisim, 0)], 0).astype(np.int8)


def fiyat_adimi(lo: float, hi: float, satir: int = 60, en_kucuk: float = 0.01) -> float:
    """Aralığı ~`satir` kademeye bölen yuvarlak adım (1-2-2,5-5 × 10^k); borsa adımının altına inmez."""
    aralik = hi - lo
    if not (math.isfinite(aralik) and aralik > 0):
        return en_kucuk
    ham = aralik / max(satir, 1)
    us = 10.0 ** math.floor(math.log10(ham))
    for k in (1.0, 2.0, 2.5, 5.0, 10.0):
        if k * us >= ham * (1 - 1e-9):
            return max(float(f"{k * us:.12g}"), en_kucuk)
    return max(10 * us, en_kucuk)


def _ondalik(adim: float) -> int:
    return max(0, int(-math.floor(math.log10(adim))) + 2) if adim > 0 else 2


def _sayi(x: float) -> float:
    return float(f"{float(x):.8g}")


def footprint(t_ms: Any, price: Any, qty: Any, side: Any, bar_ms: int, adim: float,
              oran: float = DIAGONAL_ORAN) -> dict[str, Any]:
    """İşlemlerden footprint barları + pencere hacim profili.

    Kademe = `rint(fiyat / adim)`; kademenin fiyatı `kademe × adim`. Bar başına her kademe
    `[fiyat, satış, alış, dengesizlik]` (fiyat AZALAN). Dengesizlik bit maskesidir — 1: alış[P] ≥
    oran × satış[P−1] (çapraz), 2: satış[P] ≥ oran × buy[P+1]; karşı taraf SIFIRSA işaretlenmez
    (sıfıra bölüm "sonsuz dengesizlik" değil ölçüm yokluğudur). Profil tüm pencerenin kademe
    toplamıdır; değer alanı `volume_profile._genislet` ile (aynı konvansiyon: eşitlikte AŞAĞI);
    `vah`/`val` değer alanındaki en üst/alt KADEMENİN fiyatıdır."""
    t = np.asarray(t_ms, dtype=np.int64)
    p = np.asarray(price, dtype=float)
    q = np.asarray(qty, dtype=float)
    s = np.asarray(side, dtype=np.int8)
    if t.size == 0 or bar_ms <= 0 or not adim > 0:
        return {"bars": [], "profil": None}
    dec = _ondalik(adim)
    df = pd.DataFrame({"bar": t - t % bar_ms, "lvl": np.rint(p / adim).astype(np.int64), "p": p, "q": q,
                       "buy": np.where(s > 0, q, 0.0), "sell": np.where(s < 0, q, 0.0)})
    ozet = df.groupby("bar", sort=True).agg(o=("p", "first"), h=("p", "max"), l=("p", "min"), c=("p", "last"),
                                            v=("q", "sum"), buy=("buy", "sum"), sell=("sell", "sum"),
                                            n=("p", "size"))
    hucre = df.groupby(["bar", "lvl"], sort=True)[["buy", "sell", "q"]].sum()
    bars: list[dict[str, Any]] = []
    cvd = 0.0
    for b, g in hucre.groupby(level=0, sort=True):
        lv = g.index.get_level_values(1).to_numpy()
        al, sat, top = g["buy"].to_numpy(), g["sell"].to_numpy(), g["q"].to_numpy()
        sat_alt = pd.Series(sat, index=lv).reindex(lv - 1).to_numpy()   # P−1'deki satış (yoksa NaN)
        al_ust = pd.Series(al, index=lv).reindex(lv + 1).to_numpy()     # P+1'deki alış
        with np.errstate(invalid="ignore"):
            imb = ((al > 0) & (sat_alt > 0) & (al >= oran * sat_alt)).astype(int) \
                + 2 * ((sat > 0) & (al_ust > 0) & (sat >= oran * al_ust)).astype(int)
        o = ozet.loc[b]
        delta = float(o["buy"] - o["sell"])
        cvd += delta
        sira = np.argsort(-lv, kind="stable")
        bars.append({
            "t": int(b), "o": float(o["o"]), "h": float(o["h"]), "l": float(o["l"]), "c": float(o["c"]),
            "v": _sayi(o["v"]), "buy": _sayi(o["buy"]), "sell": _sayi(o["sell"]), "delta": _sayi(delta),
            "cvd": _sayi(cvd), "n": int(o["n"]), "poc": round(float(lv[int(np.argmax(top))] * adim), dec),
            "levels": [[round(float(lv[i] * adim), dec), _sayi(sat[i]), _sayi(al[i]), int(imb[i])] for i in sira],
        })
    pr = df.groupby("lvl", sort=True)[["buy", "sell", "q"]].sum()
    tam = pr.reindex(range(int(pr.index.min()), int(pr.index.max()) + 1), fill_value=0.0)
    hist = tam["q"].to_numpy()
    toplam = float(hist.sum())
    profil = None
    if toplam > 0:
        poc = int(np.argmax(hist))
        alt, ust = _genislet(hist, poc, VALUE_AREA_PCT * toplam)
        lvl0 = int(tam.index[0])
        profil = {
            "poc": round((lvl0 + poc) * adim, dec), "vah": round((lvl0 + ust) * adim, dec),
            "val": round((lvl0 + alt) * adim, dec), "toplam": _sayi(toplam),
            "deger_alani_payi": round(float(hist[alt:ust + 1].sum() / toplam), 4),
            "levels": [[round(float(k * adim), dec), _sayi(r.sell), _sayi(r.buy)]
                       for k, r in pr.iloc[::-1].iterrows() if r.q > 0],
        }
    return {"bars": bars, "profil": profil}


def baloncuklar(t_ms: Any, price: Any, qty: Any, side: Any, en_cok: int = 150,
                bosluk_ms: int = 1, en_az: float = 0.0) -> dict[str, Any]:
    """Büyük agresif emirler. Bir piyasa emri birden çok kademeyi süpürünce birden çok baskı üretir:
    ardışık, AYNI taraftan ve aralarında ≤ `bosluk_ms` olan baskılar TEK emir sayılır (fiyat = VWAP).
    Miktarı `en_az`dan küçük emirler elenir, kalanların en büyük `en_cok`u zaman sırasıyla
    `[t, fiyat, miktar, taraf]` döner; `esik` seçilen en küçüğü."""
    t = np.asarray(t_ms, dtype=np.int64)
    p = np.asarray(price, dtype=float)
    q = np.asarray(qty, dtype=float)
    s = np.asarray(side, dtype=np.int8)
    if t.size == 0 or en_cok <= 0:
        return {"items": [], "esik": None, "n_emir": 0}
    yeni = np.ones(t.size, dtype=bool)
    yeni[1:] = (s[1:] != s[:-1]) | (np.diff(t) > bosluk_ms)
    gid = np.cumsum(yeni) - 1
    n = int(gid[-1]) + 1
    qs = np.bincount(gid, weights=q, minlength=n)
    pq = np.bincount(gid, weights=p * q, minlength=n)
    t0, s0 = t[yeni], s[yeni]
    k = min(en_cok, n)
    sec = np.argpartition(qs, n - k)[n - k:] if k < n else np.arange(n)
    sec = sec[(qs[sec] > 0) & (qs[sec] >= en_az)]
    sec = sec[np.argsort(t0[sec], kind="stable")]
    items = [[int(t0[i]), _sayi(pq[i] / qs[i]), _sayi(qs[i]), int(s0[i])] for i in sec]
    return {"items": items, "esik": _sayi(qs[sec].min()) if sec.size else None, "n_emir": n}


def isi_haritasi(goruntuler: list[tuple[int, np.ndarray, np.ndarray, float, float]], t0: int, t1: int,
                 lo: float, hi: float, adim: float, en_cok_sutun: int = 240) -> dict[str, Any] | None:
    """Defter görüntülerinden [zaman × kademe] bekleyen miktar ızgarası (alış + satış birlikte).

    Satır kademesi footprint'le AYNI eşlemedir (`rint(fiyat / adim)`), satır 0 = `rint(lo / adim)`.
    `en_cok_sutun`dan fazla görüntü eşit aralıkla SEÇİLİR (ortalanmaz)."""
    sec = [g for g in goruntuler if t0 <= g[0] <= t1]
    if not sec or not adim > 0 or not hi > lo:
        return None
    if len(sec) > en_cok_sutun:
        sec = [sec[i] for i in np.unique(np.linspace(0, len(sec) - 1, en_cok_sutun).round().astype(int))]
    k0, k1 = int(np.rint(lo / adim)), int(np.rint(hi / adim))
    satir = k1 - k0 + 1
    tablo = np.zeros((len(sec), satir), dtype=float)
    for j, (_, fp, fq, _, _) in enumerate(sec):
        r = np.rint(fp.astype(float) / adim).astype(np.int64) - k0
        m = (r >= 0) & (r < satir)
        np.add.at(tablo[j], r[m], fq[m])
    dec = _ondalik(adim)
    return {
        "t": [int(g[0]) for g in sec], "fiyat0": round(k0 * adim, dec), "adim": adim, "satir": satir,
        "degerler": np.round(tablo, 6).tolist(),
        "bid": [_sayi(g[3]) if math.isfinite(g[3]) else None for g in sec],
        "ask": [_sayi(g[4]) if math.isfinite(g[4]) else None for g in sec],
    }


# ---------------------------------------------------------------------------
# Kripto: Binance kaydedicisi (defter görüntüleri + işlemler)
# ---------------------------------------------------------------------------
class CryptoRecorder:
    """Tek sembol için arka plan kaydedicisi. Ağırlık: depth(1000)=50 / 3 sn + aggTrades=4 / 3 sn ≈
    1.080/dk (Binance sınırı 6.000/dk/IP; en çok `EN_COK_KAYDEDICI` eşzamanlı)."""

    def __init__(self, symbol: str, *, get: Callable[[str, dict[str, Any]], Any] | None = None,
                 aralik_sn: float = 3.0, pencere_dk: int = 45, isitma_dk: int = 60, sakla_dk: int = EN_COK_DAKIKA,
                 bosta_sn: float = 600.0, derinlik: int = 1000, en_cok_sayfa: int = 150) -> None:
        self.symbol = symbol
        self._get = get or _binance_get
        self.aralik_sn, self.isitma_dk, self.sakla_dk = aralik_sn, isitma_dk, sakla_dk
        self.bosta_sn, self.derinlik, self.en_cok_sayfa = bosta_sn, derinlik, en_cok_sayfa
        self._goruntu: deque[tuple[int, np.ndarray, np.ndarray, float, float]] = deque(
            maxlen=max(1, int(pencere_dk * 60 / aralik_sn)))
        self._t = np.zeros(0, dtype=np.int64)
        self._p = np.zeros(0, dtype=float)
        self._q = np.zeros(0, dtype=float)
        self._s = np.zeros(0, dtype=np.int8)
        self._son_id: int | None = None
        self._kilit = threading.Lock()
        self.hazir = threading.Event()
        self._dur = threading.Event()
        self.durdu = False
        self.hata: str | None = None
        self.son_erisim = time.monotonic()
        self.baslangic_ms = _simdi_ms()

    # --- dış arayüz ---
    def baslat(self) -> None:
        threading.Thread(target=self._calis, name=f"orderflow-{self.symbol}", daemon=True).start()

    def dur(self) -> None:
        self._dur.set()

    def islemler(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._kilit:
            return self._t.copy(), self._p.copy(), self._q.copy(), self._s.copy()

    def goruntuler(self) -> list[tuple[int, np.ndarray, np.ndarray, float, float]]:
        with self._kilit:
            return list(self._goruntu)

    # --- iç ---
    def _ekle(self, kayit: list[dict[str, Any]]) -> None:
        if self._son_id is not None:
            kayit = [x for x in kayit if int(x["a"]) > self._son_id]
        if not kayit:
            return
        t = np.fromiter((int(x["T"]) for x in kayit), dtype=np.int64, count=len(kayit))
        p = np.fromiter((float(x["p"]) for x in kayit), dtype=float, count=len(kayit))
        q = np.fromiter((float(x["q"]) for x in kayit), dtype=float, count=len(kayit))
        s = np.fromiter((-1 if x["m"] else 1 for x in kayit), dtype=np.int8, count=len(kayit))
        sinir = _simdi_ms() - self.sakla_dk * 60_000
        with self._kilit:
            self._son_id = int(kayit[-1]["a"])
            tt, pp = np.concatenate([self._t, t]), np.concatenate([self._p, p])
            qq, ss = np.concatenate([self._q, q]), np.concatenate([self._s, s])
            m = tt >= sinir
            self._t, self._p, self._q, self._s = tt[m], pp[m], qq[m], ss[m]

    def _isit(self) -> None:
        """Son `isitma_dk`yı GERİYE doğru doldur: en yeni sayfa önce, sonra `fromId` ile geri.

        aggTrade kimlikleri ARDIŞIKTIR → geri sayfaların başları önceden bilinir ve 16'şarlı PARALEL
        çekilir (ölçüldü: BTC 60 dk ≈ 58 bin işlem, sıralı 23 sn; ağırlık sayfa başı 4)."""
        from concurrent.futures import ThreadPoolExecutor
        sinir = _simdi_ms() - self.isitma_dk * 60_000
        sayfa = list(self._get("/api/v3/aggTrades", {"symbol": self.symbol, "limit": 1000}))
        parcalar = [sayfa] if sayfa else []
        n = 1
        with ThreadPoolExecutor(16) as ex:
            while parcalar and int(parcalar[-1][0]["T"]) > sinir and n < self.en_cok_sayfa:
                en_eski = int(parcalar[-1][0]["a"])
                ustler = [u for u in (en_eski - 1000 * i for i in range(min(16, self.en_cok_sayfa - n))) if u > 0]
                if not ustler:
                    break
                gelen = list(ex.map(lambda u: list(self._get("/api/v3/aggTrades", {
                    "symbol": self.symbol, "fromId": max(0, u - 1000), "limit": 1000})), ustler))
                n += len(ustler)
                for u, s in zip(ustler, gelen, strict=True):      # yeniden eskiye
                    s = [x for x in s if int(x["a"]) < u]
                    if not s:
                        break
                    parcalar.append(s)
                    if int(s[0]["T"]) <= sinir:
                        break
                else:
                    continue
                break
        self._ekle([x for parca in reversed(parcalar) for x in parca if int(x["T"]) >= sinir])

    def _islem_al(self) -> None:
        for _ in range(5):          # 3 sn'de >5.000 işlem = sayfa tavanı; kalan bir sonraki turda
            params: dict[str, Any] = {"symbol": self.symbol, "limit": 1000}
            if self._son_id is not None:
                params["fromId"] = self._son_id + 1
            sayfa = list(self._get("/api/v3/aggTrades", params))
            self._ekle(sayfa)
            if len(sayfa) < 1000:
                break

    def _goruntu_al(self) -> None:
        d = self._get("/api/v3/depth", {"symbol": self.symbol, "limit": self.derinlik})
        bids = np.asarray(d.get("bids") or [], dtype=float).reshape(-1, 2)
        asks = np.asarray(d.get("asks") or [], dtype=float).reshape(-1, 2)
        fp = np.concatenate([bids[:, 0], asks[:, 0]]).astype(np.float32)
        fq = np.concatenate([bids[:, 1], asks[:, 1]]).astype(np.float32)
        bb = float(bids[0, 0]) if len(bids) else math.nan
        ba = float(asks[0, 0]) if len(asks) else math.nan
        with self._kilit:
            self._goruntu.append((_simdi_ms(), fp, fq, bb, ba))

    def _hata_yaz(self, mesaj: str) -> None:
        if self.hata is None:
            logger.warning(f"[orderflow] {self.symbol}: {mesaj}")
        self.hata = mesaj

    def _calis(self) -> None:
        try:
            self._isit()
        except Exception as e:
            self._hata_yaz(f"ısıtma: {type(e).__name__}: {e}")
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 400:
                self.durdu = True               # geçersiz sembol: yoklamaya devam etmek boşa ağırlık
                self.hazir.set()
                return
        finally:
            self.hazir.set()
        bekle = self.aralik_sn
        while not self._dur.is_set() and time.monotonic() - self.son_erisim < self.bosta_sn:
            bas = time.monotonic()
            try:
                self._goruntu_al()
                self._islem_al()
                self.hata, bekle = None, self.aralik_sn
            except BinanceSinir as e:
                self._hata_yaz(str(e))
                bekle = e.bekle
            except Exception as e:
                self._hata_yaz(f"{type(e).__name__}: {e}")
                bekle = min(60.0, bekle * 2)
            self._dur.wait(max(0.1, bekle - (time.monotonic() - bas)))
        self.durdu = True


EN_COK_KAYDEDICI = 2
_KAYDEDICILER: dict[str, CryptoRecorder] = {}
_KAYDEDICI_KILIDI = threading.Lock()


def kaydedici(symbol: str, fabrika: Callable[[str], CryptoRecorder] = CryptoRecorder) -> CryptoRecorder:
    """Sembolün kaydedicisini getir; yoksa başlat. Tavan dolarsa en uzun süredir sorulmayanı durdurur."""
    with _KAYDEDICI_KILIDI:
        for ad in [a for a, r in _KAYDEDICILER.items() if r.durdu]:
            del _KAYDEDICILER[ad]
        r = _KAYDEDICILER.get(symbol)
        if r is None:
            canli = sorted(_KAYDEDICILER.values(), key=lambda x: x.son_erisim)
            while len(canli) >= EN_COK_KAYDEDICI:
                eski = canli.pop(0)
                eski.dur()
                _KAYDEDICILER.pop(eski.symbol, None)
            r = fabrika(symbol)
            _KAYDEDICILER[symbol] = r
            r.baslat()
        r.son_erisim = time.monotonic()
        return r


# ---------------------------------------------------------------------------
# Hisse: Alpaca işlemleri (artımlı önbellek)
# ---------------------------------------------------------------------------
@dataclass
class Islemler:
    t: np.ndarray        # epoch ms (int64)
    p: np.ndarray
    q: np.ndarray
    kirpildi: bool = False   # sayfa bütçesi doldu: pencerenin EN ESKİ kısmı eksik (kalan kısım BİTİŞİK)


def _bos() -> Islemler:
    return Islemler(np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0))


def alpaca_json_islemleri(trades: list[dict[str, Any]]) -> Islemler:
    """Alpaca `/v2/stocks/{sym}/trades` JSON'u → artan zamanlı dizi; fiyatı temsil etmeyen koşullar atılır.

    Zaman damgası RFC 3339 nanosaniye dizgisidir. Sıralama ns ile YAPILIR, sonra ms'ye kırpılır: API
    `sort=desc` döndürür ve bir milisaniyede onlarca SIP baskısı olur — önce kırpıp sonra sıralamak onları
    ters sırada bırakır, tick kuralı da ters sıra görür (Tur-103 testi yakaladı). Birim dtype'a bırakılmaz
    (pandas 3'te "us"/"ns" değişebilir, BULGU-97-1)."""
    if not trades:
        return _bos()
    ns = pd.DatetimeIndex(pd.to_datetime([x["t"] for x in trades], utc=True, format="ISO8601")).as_unit("ns").asi8
    p = np.fromiter((float(x.get("p") or "nan") for x in trades), dtype=float, count=len(trades))
    q = np.fromiter((float(x.get("s") or "nan") for x in trades), dtype=float, count=len(trades))
    ok = np.fromiter((not _HARIC_KOSUL.intersection(x.get("c") or ()) for x in trades), dtype=bool, count=len(trades))
    ok &= np.isfinite(p) & np.isfinite(q) & (p > 0) & (q > 0)
    sira = np.argsort(ns[ok], kind="stable")
    return Islemler(ns[ok][sira] // 1_000_000, p[ok][sira], q[ok][sira])


ALPACA_URL = "https://data.alpaca.markets/v2/stocks/{}/{}"
_SAYFA = 10_000
_PARCA_HEDEF = 40_000       # paralel parça başına beklenen işlem (≈4 sayfa)


def _alpaca_get(symbol: str, uc: str, prm: dict[str, Any]) -> dict[str, Any]:
    import os

    from nihai.data.alpaca_provider import AlpacaProvider
    anahtar, gizli = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", "")
    if not (anahtar and gizli):
        raise RuntimeError("ALPACA_API_KEY/SECRET yok")
    AlpacaProvider._tick("get_trades")            # REST bütçesi gözlemlenebilirliği (Madde 48)
    r = _HTTP.get(ALPACA_URL.format(symbol, uc), params=prm, timeout=30,
                  headers={"APCA-API-KEY-ID": anahtar, "APCA-API-SECRET-KEY": gizli})
    if r.status_code == 429:
        raise RuntimeError("Alpaca hız sınırı (429) — biraz sonra yeniden deneyin")
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca HTTP {r.status_code}: {r.text[:160]}")
    return r.json()


def _alpaca_parca(symbol: str, bas: datetime, son: datetime, feed: str, limit: int,
                  sayfa_butcesi: int) -> tuple[list[dict[str, Any]], bool]:
    """[bas, son] işlemleri EN YENİDEN geriye (`sort=desc`), en çok `sayfa_butcesi` sayfa.
    İkinci değer: bütçe/limit doldu, daha eski işlem VAR."""
    out: list[dict[str, Any]] = []
    jeton = None
    for _ in range(max(1, sayfa_butcesi)):
        prm: dict[str, Any] = {"start": bas.isoformat(), "end": son.isoformat(), "feed": feed, "sort": "desc",
                               "limit": min(_SAYFA, limit - len(out))}
        if jeton:
            prm["page_token"] = jeton
        j = _alpaca_get(symbol, "trades", prm)
        out.extend(j.get("trades") or [])
        jeton = j.get("next_page_token")
        if not jeton or len(out) >= limit:
            break
    return out, bool(jeton)


def parca_plani(sayimlar: list[tuple[datetime, int]], bas: datetime, son: datetime, limit: int,
                hedef: int = _PARCA_HEDEF, en_cok: int = 8) -> tuple[list[tuple[datetime, datetime, int]], bool]:
    """Dakika işlem sayılarından (1 dk barların `n`i) paralel çekim planı.

    En YENİDEN geriye `limit`e kadar sayılır (sığmayan en eski dakikalar kesilir, ikinci değer True);
    kalan aralık ~eşit işlem sayılı en çok `en_cok` parçaya bölünür → `[(bas, son, beklenen_n)]`."""
    toplam, kesim, kirp = 0, bas, False
    for t, n in reversed(sayimlar):
        if toplam + n > limit:
            kesim, kirp = t + timedelta(minutes=1), True
            break
        toplam += n
    icerde = [(t, n) for t, n in sayimlar if t >= kesim]
    k = min(en_cok, max(1, math.ceil(toplam / hedef)))
    sinirlar, birikim, j = [kesim], 0, 1
    for t, n in icerde:
        if j < k and birikim >= toplam * j / k and t > sinirlar[-1]:
            sinirlar.append(t)
            j += 1
        birikim += n
    sinirlar.append(son)
    plan = []
    for a, b in zip(sinirlar[:-1], sinirlar[1:], strict=True):
        plan.append((a, b, sum(n for t, n in icerde if a <= t < b)))
    return plan, kirp


def _alpaca_cek(symbol: str, bas: datetime, son: datetime, feed: str, limit: int) -> Islemler:
    """Alpaca REST (alpaca-py DEĞİL: NVDA 60 dk 282 bin işlemde pydantic nesnesi kurmak 13 sn).

    Önce 1 dk barlarının işlem sayısıyla (tek istek) plan kurulur: `limit`e sığan EN YENİ bitişik kesit,
    ~eşit sayılı parçalara bölünüp PARALEL çekilir (NVDA 60 dk: 3,5 sn). Toplam sayfa ≈ `limit/10.000` —
    Alpaca'nın dakikalık bütçesi hub'la PAYLAŞILIR (Tur-82: hub 429'da yfinance yedeğine düşer). Tahmin
    şaşar da bir parça bütçesini doldurursa o parçanın en eski işleminden eskisi atılır → sonuç BİTİŞİK."""
    from concurrent.futures import ThreadPoolExecutor

    from nihai.data.alpaca_provider import _to_alpaca_symbol
    sym = _to_alpaca_symbol(symbol)
    if limit < _SAYFA or son - bas <= timedelta(minutes=10):
        # çapa sorgusu (limit=1) ya da artımlı yenileme: plan isteği (1 çağrı) kazançtan pahalı
        parca, dolu = _alpaca_parca(sym, bas, son, feed, limit, math.ceil(limit / _SAYFA))
        x = alpaca_json_islemleri(parca)
        return Islemler(x.t, x.p, x.q, kirpildi=dolu)
    try:
        cubuk = _alpaca_get(sym, "bars", {"timeframe": "1Min", "start": bas.isoformat(), "end": son.isoformat(),
                                          "feed": feed, "limit": _SAYFA, "adjustment": "raw"}).get("bars") or []
        plan, kirp = parca_plani([(pd.Timestamp(b["t"]).to_pydatetime(), int(b.get("n") or 0)) for b in cubuk],
                                 bas, son, limit, hedef=_PARCA_HEDEF)
    except Exception as e:               # sayım alınamadı → tek parça, bütçe tavanıyla
        logger.debug(f"[orderflow] {sym} dakika sayımı yok ({e}) → tek parça")
        plan, kirp = [(bas, son, limit)], False
    with ThreadPoolExecutor(len(plan)) as ex:
        parcalar = list(ex.map(
            lambda i: _alpaca_parca(sym, plan[i][0] + timedelta(microseconds=1 if i else 0), plan[i][1], feed, limit,
                                    math.ceil(plan[i][2] * 1.25 / _SAYFA) + 1), range(len(plan))))
    x = alpaca_json_islemleri([tr for parca, _ in parcalar for tr in parca])
    tasan = [parca for parca, dolu in parcalar if dolu and parca]
    if not tasan:
        return Islemler(x.t, x.p, x.q, kirpildi=kirp)
    # sort=desc → parçanın SON elemanı tuttuğu en eski işlemdir (Timestamp.value her zaman ns'dir)
    m = x.t >= max(pd.Timestamp(parca[-1]["t"]).value // 1_000_000 for parca in tasan)
    return Islemler(x.t[m], x.p[m], x.q[m], kirpildi=True)


@dataclass
class _Depo:
    veri: Islemler = field(default_factory=_bos)
    bas: datetime | None = None
    son: datetime | None = None
    cekim: float = 0.0
    kirpildi: bool = False
    kilit: threading.Lock = field(default_factory=threading.Lock)


_HISSE_DEPO: dict[tuple[str, str], _Depo] = {}
_HISSE_KILIT = threading.Lock()


def hisse_islemleri(symbol: str, feed: str, bas: datetime, son: datetime, *,
                    cek: Callable[[str, datetime, datetime, str, int], Islemler] = _alpaca_cek,
                    tazelik_sn: float = 10.0, en_cok: int = EN_COK_ISLEM) -> tuple[Islemler, bool]:
    """[bas, son] işlemleri. Önbellek (sembol, feed) başına ARTIMLIDIR: yalnız (önceki son, son] çekilir;
    istenen başlangıç önbellekten eskiyse ya da ara 1 saati aşarsa tam çekim. `tazelik_sn` içinde tekrar
    çekilmez. Dönen ikinci değer: tam çekim tavana (`en_cok`) takıldı mı (en ESKİLER eksik)."""
    with _HISSE_KILIT:
        d = _HISSE_DEPO.setdefault((symbol, feed), _Depo())
    with d.kilit:
        d_bas, d_son = d.bas, d.son
        if d_bas is None or d_son is None or bas < d_bas or son - d_son > timedelta(hours=1):
            yeni = cek(symbol, bas, son, feed, en_cok)
            if yeni.t.size == 0:
                # Tur-106 (BULGU-106-26): boş tam çekim (seans kapalı) depoyu EZMEZ. Ezdiğinde çapalı pencere her
                # yenilemede yeniden tam çekiliyordu: IEX'te 5 sn'de bir, hub'la paylaşılan Alpaca bütçesinden.
                return yeni, False
            d.veri = yeni
            # Kırpılmış tam çekimde de `bas` KAPSANDI sayılır: aksi hâlde her yenileme tam çekimi
            # (≈30 sayfa) tekrarlar ve hub'la paylaşılan Alpaca bütçesini yer. Eksik kısım bayrakla söylenir.
            d_bas, d_son, d.kirpildi, d.cekim = bas, son, d.veri.kirpildi, time.monotonic()
        elif son > d_son and time.monotonic() - d.cekim >= tazelik_sn:
            x = cek(symbol, d_son + timedelta(microseconds=1), son, feed, en_cok)
            v = d.veri
            d.veri = Islemler(np.concatenate([v.t, x.t]), np.concatenate([v.p, x.p]), np.concatenate([v.q, x.q]))
            d_son, d.cekim, d.kirpildi = son, time.monotonic(), d.kirpildi or x.kirpildi
        sinir = d_son - timedelta(minutes=EN_COK_DAKIKA)       # pencere değiştirmeye yetecek kadar tut
        if d_bas < sinir:
            m = d.veri.t >= int(sinir.timestamp() * 1000)
            d.veri, d_bas = Islemler(d.veri.t[m], d.veri.p[m], d.veri.q[m]), sinir
        d.bas, d.son = d_bas, d_son
        m = (d.veri.t >= int(bas.timestamp() * 1000)) & (d.veri.t <= int(son.timestamp() * 1000))
        return Islemler(d.veri.t[m], d.veri.p[m], d.veri.q[m]), d.kirpildi


_CAPA: dict[tuple[str, str], tuple[datetime, float]] = {}


def _son_islem_capasi(symbol: str, feed: str, son: datetime,
                      cek: Callable[[str, datetime, datetime, str, int], Islemler]) -> datetime | None:
    """Pencere boşsa (seans kapalı) son işlemin anı; 5 dk önbellekli (kapalı piyasada her yenilemede
    iki çağrı yapılmasın)."""
    k = (symbol, feed)
    if k in _CAPA and time.monotonic() - _CAPA[k][1] < 300:
        return _CAPA[k][0]
    x = cek(symbol, son - timedelta(days=5), son, feed, 1)
    if x.t.size == 0:
        return None
    capa = datetime.fromtimestamp((int(x.t[-1]) + 1) / 1000, tz=UTC)
    _CAPA[k] = (capa, time.monotonic())
    return capa


# ---------------------------------------------------------------------------
# Paket
# ---------------------------------------------------------------------------
def _paketle(t: np.ndarray, p: np.ndarray, q: np.ndarray, s: np.ndarray, bar_sn: int, adim: float,
             en_kucuk_adim: float, baloncuk: int, en_az: float = 0.0) -> dict[str, Any]:
    lo, hi = float(p.min()), float(p.max())
    if not adim > 0 or (hi - lo) / adim > 1000:      # elle adım 1000 kademeyi aşarsa otomatiğe döner
        adim = fiyat_adimi(lo, hi, 60, en_kucuk_adim)
    fp = footprint(t, p, q, s, bar_sn * 1000, adim)
    hacim = float(q.sum())
    alis, satis = float(q[s > 0].sum()), float(q[s < 0].sum())
    kalan = float(q[s == 0].sum())      # fark DEĞİL doğrudan toplam: hacim−alış−satış kayan nokta artığı bırakır
    return {
        "adim": adim, "bar_sn": bar_sn, "n_islem": int(t.size), "bars": fp["bars"], "profil": fp["profil"],
        "baloncuk": baloncuklar(t, p, q, s, baloncuk, en_az=en_az),
        "hacim": {"toplam": _sayi(hacim), "alis": _sayi(alis), "satis": _sayi(satis),
                  "siniflanamayan": _sayi(kalan)},
    }


def _hisse_akisi(symbol: str, dakika: int, bar_sn: int, feed: str, adim: float, baloncuk: int,
                 simdi: datetime, cek: Callable[[str, datetime, datetime, str, int], Islemler],
                 en_az: float = 0.0) -> dict[str, Any]:
    son = simdi - SIP_GECIKME if feed == "sip" else simdi
    bas = son - timedelta(minutes=dakika)
    x, kirpildi = hisse_islemleri(symbol, feed, bas, son, cek=cek)
    capa = False
    if x.t.size == 0:
        c = _son_islem_capasi(symbol, feed, son, cek)
        if c is None:
            return {"ok": False, "reason": f"{symbol}: son 5 günde {feed.upper()} işlemi yok (sembol doğru mu?)"}
        son, bas, capa = c, c - timedelta(minutes=dakika), True
        x, kirpildi = hisse_islemleri(symbol, feed, bas, son, cek=cek)
        if x.t.size == 0:
            return {"ok": False, "reason": f"{symbol}: pencerede işlem yok"}
    en_kucuk = 0.01 if float(x.p.max()) >= 1.0 else 0.0001
    paket = _paketle(x.t, x.p, x.q, tick_kurali(x.p), bar_sn, adim, en_kucuk, baloncuk, en_az)
    kaynak = ("Alpaca SIP — tüm ABD piyasası, 15 dk GECİKMELİ" if feed == "sip"
              else "Alpaca IEX — canlı, konsolide hacmin yalnız ~%2-4'ü")
    uyari = []
    if capa:
        uyari.append("Pencere boştu (seans kapalı) → son işleme çapalandı")
    if kirpildi:
        uyari.append(f"İşlem tavanı ({EN_COK_ISLEM:,}) doldu: pencerenin EN ESKİ kısmı eksik")
    paket.update({
        "ok": True, "symbol": symbol, "piyasa": "hisse", "feed": feed, "kaynak": kaynak,
        "taraf_yontemi": "tick kuralı (TAHMİN — ABD borsaları agresör tarafını bildirmez)",
        "bas": int(bas.timestamp() * 1000), "son": int(son.timestamp() * 1000),
        "isi": None, "isi_neden": ("ABD hisseleri için ücretsiz emir defteri derinliği (L2) yok; Alpaca/IEX yalnız "
                                  "en iyi alış/satış kotasyonunu verir. Isı haritası kripto (…USDT) sembollerinde çalışır."),
        # çapalı pencere değişmez: yenileme yalnız "seans açıldı mı?" yoklamasıdır → dakikada bir yeter
        "uyari": uyari, "yenile_sn": 60 if capa else (30 if feed == "sip" else 5),
    })
    return paket


def _kripto_akisi(symbol: str, dakika: int, bar_sn: int, adim: float, baloncuk: int,
                  kaydedici_al: Callable[[str], CryptoRecorder], en_az: float = 0.0) -> dict[str, Any]:
    r = kaydedici_al(symbol)
    if not r.hazir.wait(25):
        return {"ok": False, "reason": f"{symbol}: kaydedici ısınıyor (son 60 dk işlemleri çekiliyor)",
                "yeniden_dene_sn": 3}
    t, p, q, s = r.islemler()
    if t.size == 0:
        return {"ok": False, "reason": f"{symbol}: işlem yok — {r.hata or 'Binance sembolü doğru mu?'}"}
    son = int(t[-1]) + 1
    bas = son - dakika * 60_000
    m = t >= bas
    paket = _paketle(t[m], p[m], q[m], s[m], bar_sn, adim, 1e-8, baloncuk, en_az)
    lo, hi = float(p[m].min()), float(p[m].max())
    pay = max((hi - lo) * 0.15, paket["adim"] * 5)
    goruntu = r.goruntuler()
    # Isı penceresi son İŞLEMDE değil son GÖRÜNTÜDE biter: defter, son işlemden sonra da alınır.
    isi = isi_haritasi(goruntu, bas, max(son, goruntu[-1][0]) if goruntu else son, lo - pay, hi + pay, paket["adim"])
    paket.update({
        "ok": True, "symbol": symbol, "piyasa": "kripto", "feed": "binance", "bas": bas, "son": son,
        "kaynak": "Binance spot — aggTrades + emir defteri (1000 kademe), anahtarsız, canlı",
        "taraf_yontemi": "borsa bildirimi (GERÇEK agresör tarafı)",
        "isi": isi,
        "isi_neden": None if isi else "Defter kaydı yeni başladı — ısı haritası birikiyor",
        "kaydedici": {"basladi": r.baslangic_ms, "goruntu": len(goruntu), "aralik_sn": r.aralik_sn,
                      "hata": r.hata, "isitma_dk": r.isitma_dk},
        "uyari": ["Isı haritası kaydedicinin başladığı andan itibaren birikir (Binance geçmiş defter vermez)"],
        "yenile_sn": 3,
    })
    return paket


def emir_akisi(symbol: str, *, dakika: int = 60, bar_sn: int = 60, feed: str = "sip", adim: float = 0.0,
               baloncuk: int = 150, en_az: float = 0.0, simdi: datetime | None = None,
               cek: Callable[[str, datetime, datetime, str, int], Islemler] = _alpaca_cek,
               kaydedici_al: Callable[[str], CryptoRecorder] = kaydedici) -> dict[str, Any]:
    """Panonun tek ucu: footprint barları + profil + baloncuklar (+ kriptoda ısı haritası)."""
    kripto = kripto_sembolu(symbol)
    if kripto:
        return _kripto_akisi(kripto, dakika, bar_sn, adim, baloncuk, kaydedici_al, en_az)
    return _hisse_akisi(symbol.strip().upper(), dakika, bar_sn, feed, adim, baloncuk,
                        simdi or datetime.now(UTC), cek, en_az)



# ---------------------------------------------------------------------------
# Net gamma pozlaması (GEX) — çok vadeli toplam (Tur-104)
# ---------------------------------------------------------------------------
GEX_SPAN = 0.15        # ızgara spot ±%15 (option_flow varsayılanı)
GEX_IZGARA = 121       # ±%15 / 120 aralık = %0,25 çözünürlük → flip doğrusal ara değerlemeyle
BYBIT_URL = "https://api.bybit.com"
OKX_URL = "https://www.okx.com"
_AY = {a: i + 1 for i, a in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                      "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}
_BYBIT_OPS = re.compile(r"([A-Z]+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:\.\d+)?)-([CP])(?:-[A-Z]+)?")
_OKX_OPS = re.compile(r"([A-Z]+)-USD-(\d{2})(\d{2})(\d{2})-(\d+(?:\.\d+)?)-([CP])")


def _json_get(url: str, params: dict[str, Any]) -> Any:
    r = _HTTP.get(url, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _ya_da_nan(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def kripto_tabani(symbol: str) -> str | None:
    """'BTCUSDT' / 'BTC-USD' → 'BTC' (opsiyon borsalarındaki baz coin); hisse → None."""
    s = kripto_sembolu(symbol)
    return next((s[: -len(k)] for k in _KRIPTO_KOT if s.endswith(k)), None) if s else None


def gex_temizle(taraf: Any) -> tuple[dict[str, np.ndarray] | None, int]:
    """Zincir tarafını GEX'e girebilen satırlara indir: sonlu strike>0, sonlu OI≥0, sonlu IV>0.

    `dealer_gamma_exposure` tek bir IV'siz satırda bile flip ÖLÇMEZ (sabit gammayı spot boyunca
    taşımak ölçüm değildir) → çok vadeli toplamda tek bozuk satır bütün flip'i düşürmesin diye o
    satır atılır ve SAYILIR (ikinci değer). Atılan satır sıfırla doldurulmaz."""
    s = chain_side(taraf)
    if s is None:
        return None, 0
    k = s["strike"]
    oi = s.get("oi", np.full(k.size, np.nan))
    iv = s.get("iv", np.full(k.size, np.nan))
    m = np.isfinite(k) & (k > 0) & np.isfinite(oi) & (oi >= 0) & np.isfinite(iv) & (iv > 0)
    n_atilan = int(k.size - np.count_nonzero(m))
    if not np.any(m):
        return None, n_atilan
    return {"strike": k[m], "openInterest": oi[m], "impliedVolatility": iv[m]}, n_atilan


def gex_flip(x: Any, y: Any, spot: float) -> float | None:
    """Net GEX eğrisinin spota en yakın işaret değişimi (ızgara aralığında doğrusal ara değerleme)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i = np.where((np.signbit(y[:-1]) != np.signbit(y[1:])) & (y[1:] != y[:-1]))[0]
    if i.size == 0:
        return None
    kok = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return float(kok[int(np.argmin(np.abs(kok - spot)))])


def gex_toplu(vadeler: list[tuple[str, float, Any, Any]], spot: float, *, r: float = 0.0,
              kontrat: int = 100, span: float = GEX_SPAN, izgara: int = GEX_IZGARA) -> dict[str, Any] | None:
    """Çok vadeli net GEX. `vadeler` = [(etiket, t_yıl, calls, puts)].

    Her vade `option_flow.dealer_gamma_exposure` ile AYNI spot ve AYNI ızgarada hesaplanır; GEX
    doğrusal olduğundan (Σ Γ·OI·S²·0,01) strike ve ızgara eğrileri doğrudan TOPLANIR. Flip = toplam
    ızgara eğrisinin kökü (`gex_flip`). Tek taraflı vade (yalnız call ya da yalnız put kalan) ATLANIR:
    nete tek tarafı yazmak işareti saptırırdı."""
    kg: dict[float, list[float]] = {}
    toplam: np.ndarray | None = None
    izgara_s: np.ndarray | None = None
    ozet: list[dict[str, Any]] = []
    atilan = 0
    for etiket, t, calls, puts in vadeler:
        c, dc = gex_temizle(calls)
        p, dp = gex_temizle(puts)
        atilan += dc + dp
        g = dealer_gamma_exposure(c, p, spot, t, r, kontrat, izgara, span) if (
            c is not None and p is not None and t > 0) else None
        if g is None:
            ozet.append({"vade": etiket, "atlandi": True})
            continue
        for k, a, b in zip(g["strikes"], g["call_gex"], g["put_gex"], strict=True):
            v = kg.setdefault(float(k), [0.0, 0.0])
            v[0] += float(a)
            v[1] += float(b)
        toplam = g["grid_net_gex"] if toplam is None else toplam + g["grid_net_gex"]
        izgara_s = g["grid_spot"]
        ozet.append({"vade": etiket, "t": round(float(t), 6), "net": _sayi(g["net_gex"]),
                     "flip": g["gamma_flip"], "n_strike": g["n_strikes"]})
    if toplam is None or izgara_s is None:
        return None
    ks = np.array(sorted(kg))
    cg = np.array([kg[k][0] for k in ks])
    pg = np.array([kg[k][1] for k in ks])
    net = float(np.sum(cg + pg))
    return {
        "spot": _sayi(spot), "net": _sayi(net),
        "rejim": "pozitif" if net > 0 else ("negatif" if net < 0 else "nötr"),
        "flip": gex_flip(izgara_s, toplam, spot),
        "call_wall": float(ks[int(np.argmax(cg))]) if np.any(cg > 0) else None,
        "put_wall": float(ks[int(np.argmin(pg))]) if np.any(pg < 0) else None,
        "strikes": [[float(k), _sayi(a), _sayi(b)] for k, a, b in zip(ks, cg, pg, strict=True)],
        "izgara": [[_sayi(x), _sayi(y)] for x, y in zip(izgara_s, toplam, strict=True)],
        "vadeler": ozet, "n_atilan": atilan,
        "birim": "USD / dayanağın %1 hareketi",
        "konvansiyon": "dealer call'da UZUN, put'ta KISA (VARSAYIM — gerçek pozisyon bilinmez)",
    }


def _ops_ekle(sat: dict[tuple[str, datetime], dict[str, list[list[float]]]], borsa: str, vade: datetime,
              tur: str, k: float, oi: Any, iv: Any) -> None:
    sat.setdefault((borsa, vade), {"C": [], "P": []})[tur].append([k, _ya_da_nan(oi), _ya_da_nan(iv)])


def kripto_opsiyon_vadeleri(coin: str, simdi: datetime, get: Callable[[str, dict[str, Any]], Any] = _json_get
                            ) -> tuple[float | None, list[tuple[str, float, Any, Any]], list[str], list[str]]:
    """Bybit + OKX opsiyon zincirleri → (spot, vadeler, kaynaklar, uyarılar). Anahtarsız uçlar.

    OI coin cinsinden (Bybit `openInterest`, OKX `oiCcy`) → kontrat çarpanı 1. IV = borsanın mark IV'si.
    Vade 08:00 UTC (iki borsanın da teslim saati). Deribit bu ağdan erişilemiyor (ölçüldü, 2026-10-01).
    Borsa × vade ayrı zincirdir: GEX toplamsal olduğundan iki borsanın aynı vadesi toplamda birleşir."""
    sat: dict[tuple[str, datetime], dict[str, list[list[float]]]] = {}
    kaynak: list[str] = []
    uyari: list[str] = []
    spotlar: list[float] = []
    try:
        lst = get(BYBIT_URL + "/v5/market/tickers", {"category": "option", "baseCoin": coin})["result"]["list"]
        n = 0
        for x in lst:
            m = _BYBIT_OPS.fullmatch(str(x.get("symbol", "")))
            if not m or m[1] != coin or m[3] not in _AY:
                continue
            vade = datetime(2000 + int(m[4]), _AY[m[3]], int(m[2]), 8, tzinfo=UTC)
            _ops_ekle(sat, "Bybit", vade, m[6], float(m[5]), x.get("openInterest"), x.get("markIv"))
            n += 1
            ix = _ya_da_nan(x.get("indexPrice"))
            if ix > 0:
                spotlar.append(ix)
        if n:
            kaynak.append(f"Bybit ({n} kontrat)")
    except Exception as e:  # noqa: BLE001 — borsa başına fail-soft: biri düşerse diğeri sürer
        uyari.append(f"Bybit okunamadı: {type(e).__name__}")
    try:
        aile = f"{coin}-USD"
        oi = {str(x.get("instId")): x.get("oiCcy") for x in
              get(OKX_URL + "/api/v5/public/open-interest", {"instType": "OPTION", "instFamily": aile})["data"]}
        n = 0
        for x in get(OKX_URL + "/api/v5/public/opt-summary", {"instFamily": aile})["data"]:
            m = _OKX_OPS.fullmatch(str(x.get("instId", "")))
            if not m or m[1] != coin:
                continue
            vade = datetime(2000 + int(m[2]), int(m[3]), int(m[4]), 8, tzinfo=UTC)
            _ops_ekle(sat, "OKX", vade, m[6], float(m[5]), oi.get(m[0]), x.get("markVol"))
            n += 1
        if n:
            kaynak.append(f"OKX ({n} kontrat)")
            if not spotlar:
                px = get(OKX_URL + "/api/v5/market/index-tickers", {"instId": aile})["data"]
                ix = _ya_da_nan(px[0].get("idxPx")) if px else math.nan
                if ix > 0:
                    spotlar.append(ix)
    except Exception as e:  # noqa: BLE001
        uyari.append(f"OKX okunamadı: {type(e).__name__}")
    spot = float(np.median(spotlar)) if spotlar else None
    vadeler: list[tuple[str, float, Any, Any]] = []
    for (borsa, vade), d in sorted(sat.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        t = (vade - simdi).total_seconds() / (365 * 24 * 3600)
        if t <= 0:
            continue
        c, p = (np.array(d[x], dtype=float).reshape(-1, 3) for x in ("C", "P"))
        vadeler.append((f"{borsa} {vade:%Y-%m-%d}", t,
                        {"strike": c[:, 0], "openInterest": c[:, 1], "impliedVolatility": c[:, 2]},
                        {"strike": p[:, 0], "openInterest": p[:, 1], "impliedVolatility": p[:, 2]}))
    return spot, vadeler, kaynak, uyari


def kripto_gex(coin: str, *, simdi: datetime | None = None,
               get: Callable[[str, dict[str, Any]], Any] = _json_get) -> dict[str, Any]:
    """Kripto net GEX paketi (Bybit + OKX, tüm listelenen vadeler)."""
    spot, vadeler, kaynak, uyari = kripto_opsiyon_vadeleri(coin, simdi or datetime.now(UTC), get)
    if spot is None or not vadeler:
        neden = "; ".join(uyari) or "Bybit/OKX bu coin için opsiyon listelemiyor"
        return {"ok": False, "reason": f"{coin}: opsiyon zinciri yok ({neden})"}
    g = gex_toplu(vadeler, spot, r=0.0, kontrat=1)
    if g is None:
        return {"ok": False, "reason": f"{coin}: hiçbir vadede IV + OI iki tarafta birlikte yok"}
    g.update({"ok": True, "piyasa": "kripto", "kaynak": " + ".join(kaynak), "uyari": uyari,
              "vade_ufku": "tüm listelenen vadeler", "yenile_sn": 120})
    return g
