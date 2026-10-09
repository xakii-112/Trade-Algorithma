"""Tur-103 — Emir akışı: footprint · hacim profili · baloncuklar · likidite ısı haritası.

Dört soru kilitlenir:
  1. HESAP DOĞRU MU — tick kuralı, kademe eşlemesi, korunum (hücre toplamları = bar = profil = pencere),
     çapraz dengesizlik, değer alanı konvansiyonu, baloncuk kümeleme, ısı ızgarası.
  2. VERİ YOLU DOĞRU MU — Alpaca JSON → ms (dtype biriminden bağımsız), koşul süzgeci, paralel parça planı
     (sınırda çift/eksik yok, bütçe dolunca BİTİŞİK en yeni kesit), artımlı önbellek, kapalı piyasa çapası,
     Binance geri doldurma (ardışık kimlik, paralel sayfa) ve kaydedici tavanı.
  3. FAIL-SOFT + ANAYASA — uç 500 atmaz; `def` (async değil); signals/risk/execution modülü import etmez.
  4. PANO BAĞLI MI — sekme, panel, modül kaydı, sekme kancası, Hisse Detay düğmesi; çizici gerçek biçimli JSON'u
     hatasız çizer (node varsa).
Ağ YOK: her dış çağrı enjekte edilen sahtelerle.
"""
from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.ai.volume_profile import VALUE_AREA_PCT, _genislet
from nihai.data import orderflow as O

KOK = Path(__file__).resolve().parents[1]
HTML = (KOK / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
NODE = shutil.which("node")


def _rastgele(n=4000, seed=7, bar_ms=60_000, sure_ms=30 * 60_000):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(1_790_000_000_000, 1_790_000_000_000 + sure_ms, n))
    p = np.round(100 + np.cumsum(rng.normal(0, 0.02, n)), 2)
    q = rng.integers(1, 500, n).astype(float)
    s = O.tick_kurali(p)
    return t, p, q, s


# --------------------------------------------------------------------------- 1. hesap
def test_tick_kurali_sifir_tik_oncekini_tasir_ilk_degisimden_once_siniflanamaz():
    p = np.array([10.0, 10.0, 11.0, 11.0, 10.5, 10.5, 12.0, 12.0])
    assert O.tick_kurali(p).tolist() == [0, 0, 1, 1, -1, -1, 1, 1]
    assert O.tick_kurali(np.array([])).size == 0


def test_fiyat_adimi_yuvarlak_ve_borsa_adiminin_altina_inmez():
    assert O.fiyat_adimi(100.0, 106.0, 60) == pytest.approx(0.1)
    assert O.fiyat_adimi(340.0, 342.3, 60) == pytest.approx(0.05)
    assert O.fiyat_adimi(83_000, 84_200, 60) == pytest.approx(20.0)
    assert O.fiyat_adimi(100.0, 100.1, 60, en_kucuk=0.01) == 0.01          # 0,002 → borsa adımı
    assert O.fiyat_adimi(5.0, 5.0, 60, en_kucuk=0.01) == 0.01             # dejenere aralık


def test_footprint_korunum_bar_kademe_profil_ve_cvd():
    t, p, q, s = _rastgele()
    adim = 0.05
    fp = O.footprint(t, p, q, s, 60_000, adim)
    bars = fp["bars"]
    assert sum(b["buy"] for b in bars) == pytest.approx(q[s > 0].sum())
    assert sum(b["sell"] for b in bars) == pytest.approx(q[s < 0].sum())
    assert sum(b["v"] for b in bars) == pytest.approx(q.sum())
    cvd = 0.0
    for b in bars:
        assert sum(L[2] for L in b["levels"]) == pytest.approx(b["buy"])
        assert sum(L[1] for L in b["levels"]) == pytest.approx(b["sell"])
        fiyatlar = [L[0] for L in b["levels"]]
        assert fiyatlar == sorted(fiyatlar, reverse=True)                  # fiyat AZALAN
        for f in fiyatlar:
            assert abs(f / adim - round(f / adim)) < 1e-6                  # kademe = adımın katı
        m = (t - t % 60_000) == b["t"]
        assert (b["o"], b["h"], b["l"], b["c"]) == (p[m][0], p[m].max(), p[m].min(), p[m][-1])
        cvd += b["delta"]
        assert b["cvd"] == pytest.approx(cvd, abs=1e-6)
    pr = fp["profil"]
    assert sum(L[1] + L[2] for L in pr["levels"]) == pytest.approx(q[s != 0].sum())
    assert pr["toplam"] == pytest.approx(q.sum())


def test_footprint_kademe_esleme_rint():
    t = np.array([0, 1, 2], dtype=np.int64)
    p = np.array([100.024, 100.026, 100.074])
    fp = O.footprint(t, p, np.ones(3), np.array([1, 1, 1]), 60_000, 0.05)
    assert [L[0] for L in fp["bars"][0]["levels"]] == [100.05, 100.0]      # 100.026→100.05, 100.024→100.00


def test_footprint_capraz_dengesizlik_ve_sifir_karsi_taraf_isaretlenmez():
    # kademe 100.00: satış 10 · 100.01: alış 30 (30 ≥ 3×10 → alış dengesizliği)
    # kademe 100.02: satış 30 · 100.03: alış 9  (30 ≥ 3×9 → satış dengesizliği)
    # kademe 100.05: alış 50, altındaki 100.04'te satış SIFIR (kademe var, yalnız alış) → işaret YOK
    # kademe 100.07: satış 40, üstündeki 100.08'de alış SIFIR (yalnız satış) → işaret YOK
    p = np.array([100.00, 100.01, 100.02, 100.03, 100.04, 100.05, 100.07, 100.08])
    q = np.array([10.0, 30.0, 30.0, 9.0, 1.0, 50.0, 40.0, 2.0])
    s = np.array([-1, 1, -1, 1, 1, 1, -1, -1])
    fp = O.footprint(np.arange(8, dtype=np.int64), p, q, s, 60_000, 0.01)
    imb = {L[0]: L[3] for L in fp["bars"][0]["levels"]}
    assert imb[100.01] & 1 and not imb[100.01] & 2
    assert imb[100.02] & 2
    assert imb[100.05] == 0 and imb[100.0] == 0 and imb[100.03] == 0 and imb[100.07] == 0
    # sınır: 29 < 3×10 → işaret yok
    fp2 = O.footprint(np.arange(2, dtype=np.int64), p[:2], np.array([10.0, 29.0]), s[:2], 60_000, 0.01)
    assert all(L[3] == 0 for L in fp2["bars"][0]["levels"])


def test_profil_deger_alani_volume_profile_konvansiyonu():
    t, p, q, s = _rastgele(seed=3)
    adim = 0.05
    pr = O.footprint(t, p, q, s, 60_000, adim)["profil"]
    lvl = np.rint(p / adim).astype(int)
    hist = np.bincount(lvl - lvl.min(), weights=q)
    poc = int(np.argmax(hist))
    alt, ust = _genislet(hist, poc, VALUE_AREA_PCT * hist.sum())
    assert pr["poc"] == pytest.approx((lvl.min() + poc) * adim)
    assert pr["val"] == pytest.approx((lvl.min() + alt) * adim)
    assert pr["vah"] == pytest.approx((lvl.min() + ust) * adim)
    assert pr["val"] <= pr["poc"] <= pr["vah"] and pr["deger_alani_payi"] >= VALUE_AREA_PCT


def test_baloncuk_supurme_birlesir_taraf_ve_bosluk_boler():
    t = np.array([1000, 1000, 1000, 1000, 1005, 2000], dtype=np.int64)
    p = np.array([10.0, 10.1, 10.2, 10.2, 10.3, 9.9])
    q = np.array([100.0, 100.0, 200.0, 5.0, 50.0, 1.0])
    s = np.array([1, 1, 1, -1, -1, -1])
    b = O.baloncuklar(t, p, q, s, en_cok=10)
    assert b["n_emir"] == 4                                   # süpürme · taraf değişimi · 5 ms boşluk · 1 sn boşluk
    ilk = b["items"][0]
    assert ilk[0] == 1000 and ilk[2] == 400.0 and ilk[3] == 1
    assert ilk[1] == pytest.approx((10.0 * 100 + 10.1 * 100 + 10.2 * 200) / 400)   # VWAP
    en_buyuk = O.baloncuklar(t, p, q, s, en_cok=2)
    assert [x[2] for x in en_buyuk["items"]] == [400.0, 50.0] and en_buyuk["esik"] == 50.0
    assert [x[0] for x in en_buyuk["items"]] == sorted(x[0] for x in en_buyuk["items"])   # zaman sırası


def test_isi_haritasi_korunum_ve_footprint_ile_ayni_kademe():
    fp = np.array([99.96, 100.0, 100.04, 100.51, 101.0], dtype=np.float32)
    fq = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    g = [(1000, fp, fq, 100.0, 100.05), (4000, fp, fq * 2, 100.0, 100.05)]
    I = O.isi_haritasi(g, 0, 10_000, 99.9, 100.6, 0.05)
    assert I["satir"] == int(np.rint(100.6 / 0.05)) - int(np.rint(99.9 / 0.05)) + 1
    k0 = int(np.rint(99.9 / 0.05))
    for j, carp in ((0, 1), (1, 2)):
        kol = I["degerler"][j]
        assert sum(kol) == pytest.approx(10.0 * carp)                  # 101.0 aralık DIŞI
        for f, m in zip(fp[:4], fq[:4], strict=True):
            assert kol[int(np.rint(float(f) / 0.05)) - k0] >= m * carp - 1e-6
    assert I["bid"] == [100.0, 100.0] and I["t"] == [1000, 4000]
    assert O.isi_haritasi(g, 5000, 9000, 99.9, 100.6, 0.05) is None    # pencerede görüntü yok
    cok = [(i, fp, fq, 1.0, 2.0) for i in range(1000)]
    I2 = O.isi_haritasi(cok, 0, 10_000, 99.9, 100.6, 0.05, en_cok_sutun=50)
    assert len(I2["t"]) <= 50 and I2["t"][0] == 0 and I2["t"][-1] == 999


# --------------------------------------------------------------------------- 2. veri yolu
def test_alpaca_json_ms_kosul_suzgeci_ve_siralama():
    tr = [
        {"t": "2026-09-28T17:25:43.775974941Z", "p": 230.35, "s": 100, "c": ["@"]},
        {"t": "2026-09-28T17:25:43.775001000Z", "p": 230.30, "s": 5, "c": ["@", "I"]},      # tek lot TUTULUR
        {"t": "2026-09-28T17:25:44.000000000Z", "p": 229.00, "s": 900, "c": ["@", "W"]},    # ortalama fiyat
        {"t": "2026-09-28T17:25:45.000000000Z", "p": 230.40, "s": 9e6, "c": ["M"]},         # resmi kapanış raporu
        {"t": "2026-09-28T17:25:46.000000000Z", "p": 0.0, "s": 10, "c": ["@"]},             # geçersiz fiyat
    ]
    x = O.alpaca_json_islemleri(tr)
    beklenen = int(pd.Timestamp("2026-09-28T17:25:43.775Z").value // 1_000_000)
    assert x.t.tolist() == [beklenen, beklenen] and x.p.tolist() == [230.30, 230.35] and x.q.tolist() == [5, 100]
    assert O.alpaca_json_islemleri([]).t.size == 0


def _dk(i):
    return datetime(2026, 9, 28, 14, 0, tzinfo=UTC) + timedelta(minutes=i)


def test_parca_plani_en_yeniden_keser_ve_esit_boler():
    sayim = [(_dk(i), 10_000) for i in range(60)]                  # 600 bin işlem
    plan, kirp = O.parca_plani(sayim, _dk(0), _dk(60), 300_000)
    assert kirp and plan[0][0] == _dk(30)                          # en yeni 30 dk sığar
    assert plan[-1][1] == _dk(60) and len(plan) == 8
    for (a, b, _), (c, _, _) in zip(plan[:-1], plan[1:], strict=True):
        assert a < b == c                                          # bitişik, sıralı
    assert sum(n for _, _, n in plan) == 300_000
    assert max(n for *_, n in plan) - min(n for *_, n in plan) <= 10_000
    plan2, kirp2 = O.parca_plani(sayim[:3], _dk(0), _dk(60), 300_000)
    assert not kirp2 and plan2[0][0] == _dk(0) and len(plan2) == 1
    assert O.parca_plani([], _dk(0), _dk(60), 300_000) == ([(_dk(0), _dk(60), 0)], False)


class _SahteAlpaca:
    """Tarih aralığı + sort=desc + sayfa jetonu + 1 dk bar sayımı taklidi (ns damgalı)."""

    def __init__(self, n=2500, sure_dk=60, seed=1, sayim_bolen=1):
        rng = np.random.default_rng(seed)
        self.sayim_bolen = sayim_bolen                                # >1: bar sayımı EKSİK (plan tahmini şaşar)
        bas = pd.Timestamp(_dk(0)).value
        self.ns = np.sort(rng.integers(bas, bas + sure_dk * 60 * 10**9, n))
        self.ns[:sure_dk] = bas + np.arange(sure_dk) * 60 * 10**9    # HER dakika başında bir işlem: parça
        self.ns.sort()                                                # sınırına tam denk gelen baskı hep var
        self.cagri = 0
        self.parca_baslari: set[str] = set()

    def __call__(self, symbol, uc, prm):
        self.cagri += 1
        if uc == "trades":
            self.parca_baslari.add(prm["start"])
        else:
            self.bar_cagri = getattr(self, "bar_cagri", 0) + 1
        a, b = pd.Timestamp(prm["start"]).value, pd.Timestamp(prm["end"]).value
        sec = self.ns[(self.ns >= a) & (self.ns <= b)]
        if uc == "bars":
            dk = pd.Series(1, index=pd.to_datetime(sec, utc=True).floor("min")).groupby(level=0).sum()
            return {"bars": [{"t": k.isoformat(), "n": int(v) // self.sayim_bolen} for k, v in dk.items()]}
        sec = sec[::-1]
        bas = int(prm.get("page_token") or 0)
        sayfa = sec[bas: bas + prm["limit"]]
        jeton = str(bas + len(sayfa)) if bas + len(sayfa) < len(sec) else None
        return {"trades": [{"t": pd.Timestamp(int(v), tz="UTC").isoformat(), "p": 100 + (int(v) % 97) / 100,
                            "s": 1, "c": ["@"]} for v in sayfa], "next_page_token": jeton}


def test_alpaca_cek_paralel_parcalar_tek_parcayla_ayni_kume(monkeypatch):
    sahte = _SahteAlpaca()
    monkeypatch.setattr(O, "_alpaca_get", sahte)
    monkeypatch.setattr(O, "_SAYFA", 100)
    monkeypatch.setattr(O, "_PARCA_HEDEF", 400)
    x = O._alpaca_cek("TEST", _dk(0), _dk(60), "sip", 100_000)
    beklenen = np.sort(sahte.ns // 1_000_000)
    assert not x.kirpildi and x.t.tolist() == beklenen.tolist()   # sınırdaki işlem ÇİFT de EKSİK de değil
    assert len(sahte.parca_baslari) >= 7                          # plan GERÇEKTEN ≥7 paralel parçaya bölündü


def test_alpaca_cek_kisa_aralikta_plan_istegi_yapmaz(monkeypatch):
    """Artımlı yenileme (≤10 dk) tek parça: bar sayımı isteği hub'la paylaşılan bütçeyi boşa harcamaz."""
    sahte = _SahteAlpaca()
    monkeypatch.setattr(O, "_alpaca_get", sahte)
    monkeypatch.setattr(O, "_SAYFA", 100)
    x = O._alpaca_cek("TEST", _dk(10), _dk(15), "sip", 100_000)
    ms = np.sort(sahte.ns // 1_000_000)
    a, b = int(_dk(10).timestamp() * 1000), int(_dk(15).timestamp() * 1000)
    assert getattr(sahte, "bar_cagri", 0) == 0 and x.t.tolist() == ms[(ms >= a) & (ms <= b)].tolist()


def test_alpaca_cek_sayim_eksikse_tasan_parcalar_bitisik_kesit_birakir(monkeypatch):
    """Bar sayımı işlemleri eksik sayarsa parça bütçeleri dolar: en yeni TAŞAN parçanın en eskisinden
    eskisi atılır → boşluklu değil BİTİŞİK kesit (kalan her şey en yeni işlemlerin kuyruğu)."""
    sahte = _SahteAlpaca(n=2500, sayim_bolen=3)
    monkeypatch.setattr(O, "_alpaca_get", sahte)
    monkeypatch.setattr(O, "_SAYFA", 100)
    monkeypatch.setattr(O, "_PARCA_HEDEF", 200)
    x = O._alpaca_cek("TEST", _dk(0), _dk(60), "sip", 100_000)
    tum = np.sort(sahte.ns // 1_000_000)
    assert x.kirpildi and 0 < x.t.size < tum.size
    assert x.t.tolist() == tum[-x.t.size:].tolist()


def test_alpaca_cek_butce_dolunca_bitisik_en_yeni_kesit(monkeypatch):
    sahte = _SahteAlpaca(n=3000)
    monkeypatch.setattr(O, "_alpaca_get", sahte)
    monkeypatch.setattr(O, "_SAYFA", 100)
    monkeypatch.setattr(O, "_PARCA_HEDEF", 400)
    x = O._alpaca_cek("TEST", _dk(0), _dk(60), "sip", 1000)
    tum = np.sort(sahte.ns // 1_000_000)
    assert x.kirpildi and 0 < x.t.size <= 1000
    assert x.t.tolist() == tum[-x.t.size:].tolist()                # en YENİ ve BİTİŞİK


def test_hisse_islemleri_artimli_tazelik_ve_tam_cekim(monkeypatch):
    O._HISSE_DEPO.clear()
    cagrilar = []

    def cek(sym, bas, son, feed, limit):
        cagrilar.append((bas, son))
        ms = np.arange(int(bas.timestamp() * 1000) + 1, int(son.timestamp() * 1000), 30_000, dtype=np.int64)
        return O.Islemler(ms, np.full(ms.size, 10.0), np.ones(ms.size))

    saat = [1000.0]
    monkeypatch.setattr(O.time, "monotonic", lambda: saat[0])
    x, _ = O.hisse_islemleri("AAA", "sip", _dk(0), _dk(60), cek=cek)
    assert cagrilar == [(_dk(0), _dk(60))] and x.t.size > 0
    O.hisse_islemleri("AAA", "sip", _dk(1), _dk(61), cek=cek)           # tazelik içinde → çağrı YOK
    assert len(cagrilar) == 1
    saat[0] += 11
    x2, _ = O.hisse_islemleri("AAA", "sip", _dk(1), _dk(61), cek=cek)
    assert cagrilar[-1] == (_dk(60) + timedelta(microseconds=1), _dk(61))   # YALNIZ yeni aralık
    assert x2.t.min() >= int(_dk(1).timestamp() * 1000) and np.all(np.diff(x2.t) > 0)
    saat[0] += 11
    O.hisse_islemleri("AAA", "sip", _dk(-30), _dk(61), cek=cek)          # daha eski başlangıç → tam
    assert cagrilar[-1] == (_dk(-30), _dk(61))
    saat[0] += 11
    O.hisse_islemleri("AAA", "sip", _dk(100), _dk(200), cek=cek)         # ara > 1 sa → tam
    assert cagrilar[-1] == (_dk(100), _dk(200))
    O._HISSE_DEPO.clear()


def test_kapali_piyasada_son_islem_capasi(monkeypatch):
    O._HISSE_DEPO.clear()
    O._CAPA.clear()
    son_islem = _dk(0)

    def cek(sym, bas, son, feed, limit):
        if limit == 1:
            return O.Islemler(np.array([int(son_islem.timestamp() * 1000)]), np.array([10.0]), np.array([1.0]))
        if son <= son_islem + timedelta(milliseconds=1):
            ms = np.arange(int(bas.timestamp() * 1000) + 1, int(son.timestamp() * 1000), 1000, dtype=np.int64)
            return O.Islemler(ms, 10 + (ms % 7) / 100, np.ones(ms.size))
        return O._bos()

    r = O.emir_akisi("AAA", dakika=30, bar_sn=60, feed="sip", simdi=_dk(3000), cek=cek)
    assert r["ok"] and any("çapalandı" in u for u in r["uyari"])
    assert r["son"] == int(son_islem.timestamp() * 1000) + 1 and r["isi"] is None and r["isi_neden"]
    O._HISSE_DEPO.clear()
    O._CAPA.clear()


class _SahteBinance:
    def __init__(self, simdi_ms, n=5000, adim_ms=1000):
        self.simdi = simdi_ms
        self.idler = np.arange(10_000, 10_000 + n)
        self.T = simdi_ms - (n - 1 - np.arange(n)) * adim_ms
        self.istek = []

    def _kayit(self, i):
        return {"a": int(self.idler[i]), "p": str(100 + i % 5), "q": "0.5", "T": int(self.T[i]), "m": bool(i % 2)}

    def __call__(self, yol, prm):
        self.istek.append((yol, dict(prm)))
        if yol.endswith("depth"):
            return {"bids": [["99.9", "2"], ["99.8", "3"]], "asks": [["100.1", "1"], ["100.2", "4"]]}
        if "fromId" in prm:
            bas = int(np.searchsorted(self.idler, prm["fromId"]))
        else:
            bas = max(0, len(self.idler) - prm["limit"])
        return [self._kayit(i) for i in range(bas, min(len(self.idler), bas + prm["limit"]))]


def test_kaydedici_geri_doldurma_pencere_tekil_sirali_ve_taraf(monkeypatch):
    simdi = 1_790_000_000_000
    monkeypatch.setattr(O, "_simdi_ms", lambda: simdi)
    sahte = _SahteBinance(simdi)
    r = O.CryptoRecorder("TESTUSDT", get=sahte, isitma_dk=30)
    r._isit()
    t, p, q, s = r.islemler()
    beklenen = sahte.T[sahte.T >= simdi - 30 * 60_000]
    assert t.tolist() == beklenen.tolist()                             # tekil, sıralı, eksiksiz
    assert set(s.tolist()) == {1, -1} and s[0] == (-1 if sahte._kayit(len(sahte.idler) - len(t))["m"] else 1)
    sayfa = [i for i in sahte.istek if "fromId" in i[1]]
    assert len(sayfa) >= 1 and all(i[1]["limit"] == 1000 for i in sayfa)
    sahte.idler = np.append(sahte.idler, [sahte.idler[-1] + 1, sahte.idler[-1] + 2])
    sahte.T = np.append(sahte.T, [simdi + 1, simdi + 2])
    r._islem_al()
    t2 = r.islemler()[0]
    assert t2.size == t.size + 2 and np.all(np.diff(t2) >= 0)          # yalnız YENİ kimlikler
    r._goruntu_al()
    (_, fp, fq, bb, ba), = r.goruntuler()
    assert (bb, ba) == (99.9, 100.1) and fq.sum() == pytest.approx(10.0)


def test_kaydedici_tavani_en_uzun_sorulmayani_durdurur():
    O._KAYDEDICILER.clear()

    class Sahte:
        def __init__(self, s):
            self.symbol, self.durdu, self.son_erisim, self.durduruldu = s, False, 0.0, False

        def baslat(self):
            pass

        def dur(self):
            self.durduruldu = True

    a = O.kaydedici("AUSDT", fabrika=Sahte)
    b = O.kaydedici("BUSDT", fabrika=Sahte)
    a.son_erisim, b.son_erisim = 5.0, 9.0
    O.kaydedici("CUSDT", fabrika=Sahte)
    assert a.durduruldu and not b.durduruldu and set(O._KAYDEDICILER) == {"BUSDT", "CUSDT"}
    O._KAYDEDICILER.clear()


def test_kripto_isi_penceresi_son_islemden_sonraki_goruntuyu_kapsar():
    """Regresyon: defter görüntüsü son işlemden SONRA alınınca ısı haritası boş kalıyordu."""
    class Kayit:
        hazir, hata, baslangic_ms, aralik_sn, isitma_dk = type("E", (), {"wait": lambda s, x: True})(), None, 0, 3.0, 60

        def islemler(self):
            t = np.arange(0, 60_000, 1000, dtype=np.int64)
            q = np.random.default_rng(5).random(t.size) * 0.37 + 0.001     # kayan nokta miktarlar
            return t, 100 + (t % 7000) / 1e5, q, np.where(t % 2000 == 0, 1, -1).astype(np.int8)

        def goruntuler(self):
            return [(59_000 + 2000, np.array([100.0, 100.05], np.float32), np.array([3.0, 4.0], np.float32), 100.0, 100.05)]

    r = O.emir_akisi("XUSDT", dakika=5, bar_sn=60, kaydedici_al=lambda s: Kayit())
    assert r["ok"] and r["isi"] is not None and len(r["isi"]["t"]) == 1
    assert r["hacim"]["siniflanamayan"] == 0.0                         # kayan nokta artığı yok (-0.0000 regresyonu)


def test_kripto_sembol_esleme():
    assert O.kripto_sembolu("btcusdt") == "BTCUSDT"
    assert O.kripto_sembolu("BTC-USD") == "BTCUSDT"
    assert O.kripto_sembolu("AAPL") is None and O.kripto_sembolu("USDT") is None and O.kripto_sembolu("BRK-B") is None


# --------------------------------------------------------------------------- 3. fail-soft + anayasa
def test_uc_fail_soft_ve_dogrulama(monkeypatch):
    from fastapi.testclient import TestClient

    from nihai.web import api_orderflow as A
    from nihai.web.web_dashboard import app
    c = TestClient(app)
    assert c.get("/api/orderflow", params={"symbol": "AA$PL"}).json()["ok"] is False
    assert c.get("/api/orderflow", params={"symbol": "AAPL", "bar": 7}).json()["ok"] is False
    assert c.get("/api/orderflow", params={"symbol": "AAPL", "feed": "otc"}).json()["ok"] is False
    assert "bar" in c.get("/api/orderflow", params={"symbol": "AAPL", "minutes": 240, "bar": 15}).json()["reason"]

    def patla(*a, **k):
        raise RuntimeError("ağ yok")
    monkeypatch.setattr(A.OF, "emir_akisi", patla)
    r = c.get("/api/orderflow", params={"symbol": "AAPL"})
    assert r.status_code == 200 and r.json() == {"ok": False, "reason": "RuntimeError: ağ yok"}
    alinan = {}
    monkeypatch.setattr(A.OF, "emir_akisi", lambda sym, **k: alinan.update(sym=sym, **k) or {"ok": True})
    c.get("/api/orderflow", params={"symbol": "nvda", "minutes": 9999, "bar": 300, "feed": "IEX", "bubbles": 9999})
    assert alinan == {"sym": "NVDA", "dakika": 240, "bar_sn": 300, "feed": "iex", "adim": 0.0, "baloncuk": 500,
                      "en_az": 0.0}
    assert not inspect.iscoroutinefunction(A.orderflow)                # thread havuzu (Tur-66)


def test_karar_yolu_emir_akisini_import_etmez():
    for alt in ("signals", "risk", "execution"):
        for py in (KOK / "nihai" / alt).rglob("*.py"):
            for d in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
                if isinstance(d, ast.ImportFrom):
                    adlar = [d.module or ""] + [f"{d.module}.{a.name}" for a in d.names]
                elif isinstance(d, ast.Import):
                    adlar = [a.name for a in d.names]
                else:
                    continue
                assert not any(a.endswith("orderflow") or "orderflow." in a for a in adlar), py


# --------------------------------------------------------------------------- 4. pano
def test_pano_sekme_panel_modul_ve_kancalar_bagli():
    assert 'aria-controls="tab-orderflow"' in HTML and '<div id="tab-orderflow" class="tab-content">' in HTML
    for id_ in ("of-sym", "of-dk", "of-bar", "of-feed", "of-canvas", "of-sarmal", "of-bilgi", "of-k-isi", "of-k-fp",
                "of-k-bal", "of-k-prof", "of-k-cvd", "of-kontrast", "of-canli"):
        assert f'id="{id_}"' in HTML, id_
    assert "if (tabId === 'tab-orderflow' && typeof ofYukle === 'function') ofYukle();" in HTML
    assert "ofAc(document.getElementById('az-symbol').value)" in HTML        # Hisse Detay → Emir Akışı
    beklenen = json.loads(re.search(r"var beklenen = (\[.*?\]);", HTML).group(1))
    assert "m11-emir-akisi" in beklenen
    assert "apiCall('orderflow?' + q.toString())" in HTML


_CIZ = r"""
const fs = require("fs"); const vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8"); const veri = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const blok = /<script>\s*\/\/ ===== \[pano modülü m11-emir-akisi\][\s\S]*?<\/script>/.exec(html)[0].replace(/<\/?script>/g, "");
const cagri = { fillRect: 0, arc: 0, fillText: 0 };
const ctx = new Proxy({}, { get(t, k) { if (k in t) return t[k]; return (...a) => { if (k in cagri) cagri[k]++; }; },
                            set(t, k, v) { t[k] = v; return true; } });
const el = (ek) => Object.assign({ checked: true, value: "50", style: {}, clientWidth: 1400, clientHeight: 660,
  width: 0, height: 0, offsetWidth: 100, offsetHeight: 40, textContent: "", innerHTML: "", classList: { contains: () => true },
  getContext: () => ctx, addEventListener() {}, getBoundingClientRect: () => ({ left: 0, top: 0 }) }, ek || {});
const dom = {};
const document = { getElementById: (id) => (dom[id] = dom[id] || el()), createElement: () => el(), querySelector: () => null };
const sb = { document, window: {}, console, Math, JSON, Number, String, Date, Array, Object, isFinite, URLSearchParams,
             setTimeout: () => 1, clearTimeout() {} };
sb.window = sb; sb.window.__PANO_MODULLERI = []; sb.window.devicePixelRatio = 1;
vm.createContext(sb); vm.runInContext(blok, sb);
vm.runInContext("OF.veri = " + JSON.stringify(veri) + "; ofBilgi(OF.veri); ofCiz(); OF.fare = {x: 300, y: 200}; ofImlec(); OF.n = 5; ofCiz();", sb);
console.log(JSON.stringify({ cagri, bilgi: dom["of-bilgi"].innerHTML.length, ipucu: dom["of-ipucu"].innerHTML,
  kisa: [sb.ofKisa(1234), sb.ofKisa(2500000), sb.ofKisa(0.5), sb.ofKisa(null)], ondalik: [sb.ofOndalik(0.05), sb.ofOndalik(2.5), sb.ofOndalik(20)],
  modul: sb.window.__PANO_MODULLERI }));
"""


@pytest.mark.skipif(NODE is None, reason="node kurulu değil — JS ölçülemedi")
def test_cizici_gercek_bicimli_json_u_hatasiz_cizer(tmp_path):
    t, p, q, s = _rastgele(n=3000)
    v = O._paketle(t, p, q, s, 60, 0.0, 0.01, 50)
    fp32 = np.array([99.5, 100.0, 100.5], dtype=np.float32)
    v.update({"ok": True, "symbol": "TEST", "feed": "sip", "kaynak": "test", "taraf_yontemi": "tick",
              "bas": int(t[0]), "son": int(t[-1]), "uyari": ["deneme"], "yenile_sn": 30, "isi_neden": None,
              "kaydedici": {"basladi": int(t[0]), "goruntu": 2, "aralik_sn": 3, "hata": None, "isitma_dk": 60},
              "isi": O.isi_haritasi([(int(t[10]), fp32, fp32 / 100, 99.9, 100.1), (int(t[-5]), fp32, fp32 / 50, 99.9, 100.1)],
                                    int(t[0]), int(t[-1]), float(p.min()), float(p.max()), v["adim"])})
    (tmp_path / "v.json").write_text(json.dumps(v), encoding="utf-8")
    (tmp_path / "c.js").write_text(_CIZ, encoding="utf-8")
    html = tmp_path / "p.html"
    html.write_text(HTML, encoding="utf-8")
    r = subprocess.run([NODE, str(tmp_path / "c.js"), str(html), str(tmp_path / "v.json")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[:800]
    o = json.loads(r.stdout)
    assert o["modul"] == ["m11-emir-akisi"]
    assert o["cagri"]["fillRect"] > 100 and o["cagri"]["arc"] > 0 and o["cagri"]["fillText"] > 10
    assert o["bilgi"] > 100 and "Hacim" in o["ipucu"]
    assert o["kisa"] == ["1.2K", "2.5M", "0.50", "—"] and o["ondalik"] == [2, 1, 0]
