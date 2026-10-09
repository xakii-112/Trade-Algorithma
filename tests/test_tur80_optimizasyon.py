"""Tur-80: ölçülmüş hızlandırmaların DENKLİK kilitleri.

Her hızlandırma eski yolun çıktısını birebir vermek ZORUNDADIR; bu dosya eski yolu referans olarak
taşır ve yeni yolu ona karşı ölçer (Tur-36: denklik varsayılmaz, ızgarada ölçülür).
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai import tax_report as tr
from nihai.execution import prediction_tracker as pt

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# vergi raporu — eski (referans) yardımcılar
# ---------------------------------------------------------------------------
def _eski_parse_ts(v):
    try:
        t = pd.to_datetime(v, errors="coerce", utc=True)
    except (TypeError, ValueError):
        return None
    if t is pd.NaT or pd.isna(t):
        return None
    return t.to_pydatetime().replace(tzinfo=None)


def _eski_wash(df):
    adaylar = set()
    for _sym, grup in df.groupby("symbol"):
        girisler = grup["_entry"].dropna().tolist()
        for idx, satir in grup.iterrows():
            if not (satir["_pnl"] < 0) or pd.isna(satir["_exit"]):
                continue
            for g in girisler:
                if g == satir["_entry"]:
                    continue
                if abs((g - satir["_exit"]).days) <= tr.WASH_SALE_WINDOW_DAYS:
                    adaylar.add(int(idx))
                    break
    return adaylar


def _eski_kaynak(df, wash):
    return [{**r, "_wash_candidate": int(idx) in wash} for idx, r in df.iterrows()]


def _tarih_izgarasi() -> list:
    izgara = ["2026-07-06T12:01:12.596472", "2026-07-06 12:01:12", "2026-07-06", "2026-07-06T12:01:12Z",
              "2026-07-06T12:01:12+03:00", "2026-07-06T12:01:12.5-04:00", "2026-07-06T12:01:12.123456789",
              "2026/07/06 12:01", "06.07.2026", "", " ", "abc", "NaT", "nan", None, float("nan"), 0,
              "2026-02-30", "20260706T120112", "2026-07-06T12:01:12.596472+00:00",
              # Codex incelemesinin karşı örnekleri: stdlib kabul eder, pandas reddeder
              "2026-W01-1", "2026-09-16X12:30:00", "2026-09-16T12:30:00+05:30:15", "2026-259",
              "2026-09-16T12", "2026-09-16T12:30:00,5", "2026-09-16 12:30:00.1234567", "2026-09-16T24:00:00",
              "2026-09-16T12:30Z", "+2026-09-16", "2026-09-16T12:30:00.5+0530",
              pd.Timestamp("2026-07-06 12:00", tz="US/Eastern"), datetime(2026, 7, 6, 12, 0)]
    rng = random.Random(80)
    for _ in range(400):
        y, mo, d = rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28)
        h, mi, s, us = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59), rng.randint(0, 999999)
        izgara.append(f"{y:04d}-{mo:02d}-{d:02d}{rng.choice('T ')}{h:02d}:{mi:02d}:{s:02d}"
                      f"{rng.choice(['', f'.{us:06d}', f'.{us // 1000:03d}'])}"
                      f"{rng.choice(['', 'Z', '+00:00', '-04:00', '+05:30'])}")
    return izgara


def test_parse_ts_hizli_yol_pandas_yoluyla_birebir():
    """ISO dizgide stdlib, ayrıştıramadığında pandas yolu — değer VE tip aynı."""
    for v in _tarih_izgarasi():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eski, yeni = _eski_parse_ts(v), tr._parse_ts(v)
        assert eski == yeni and type(eski) is type(yeni), repr(v)


def _islemler_df(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    satirlar = []
    for i in range(n):
        giris = datetime(2026, 1, 1) + pd.Timedelta(days=rng.randint(0, 200), minutes=rng.randint(0, 900))
        cikis = giris + pd.Timedelta(days=rng.randint(-2, 40))
        satir = {"exit_time": cikis.isoformat(), "symbol": rng.choice(["AAA", "BBB", "SPX", "CCC"]),
                 "asset_type": rng.choice(["stock", "stock", "option"]),
                 "side": rng.choice(["LONG", "LONG", "SHORT"]), "entry_time": giris.isoformat(),
                 "entry_price": round(rng.uniform(10, 200), 4), "exit_price": round(rng.uniform(10, 200), 4),
                 "quantity": rng.choice([1, 5, 10, 25]), "realized_pnl": round(rng.uniform(-500, 500), 2)}
        k = rng.random()
        if k < 0.05:
            satir["entry_time"] = "bozuk"
        elif k < 0.10:
            satir["exit_time"] = None
        elif k < 0.15:
            satir["exit_time"] = cikis.isoformat() + "+02:00"
        elif k < 0.18:
            satir["quantity"] = float("nan")
        satirlar.append(satir)
        _ = i
    # SEYREK uç: yoğun kurguda her zararlı işlemin 30 gün içinde BAŞKA bir girişi olur ve "kendi
    # girişini atla" kuralı hiç ayırt edilmez (mutasyon kaçtı). Tek işlemli zararlı sembol + 30 günden
    # uzak ikinci girişli sembol bu kuralı ölçer.
    satirlar.append({"exit_time": "2026-03-11T10:00:00", "symbol": "SOLO", "asset_type": "stock",
                     "side": "LONG", "entry_time": "2026-03-01T10:00:00", "entry_price": 50.0,
                     "exit_price": 45.0, "quantity": 10, "realized_pnl": -50.0})
    satirlar.append({"exit_time": "2026-04-10T10:00:00", "symbol": "DUO", "asset_type": "stock",
                     "side": "LONG", "entry_time": "2026-04-01T10:00:00", "entry_price": 50.0,
                     "exit_price": 45.0, "quantity": 10, "realized_pnl": -50.0})
    satirlar.append({"exit_time": "2026-06-20T10:00:00", "symbol": "DUO", "asset_type": "stock",
                     "side": "LONG", "entry_time": "2026-06-01T10:00:00", "entry_price": 50.0,
                     "exit_price": 55.0, "quantity": 10, "realized_pnl": 50.0})
    return pd.DataFrame(satirlar)


def _hazirla(df: pd.DataFrame, parse) -> pd.DataFrame:
    df = df.copy()
    df["_entry"] = df["entry_time"].map(parse)
    df["_exit"] = df["exit_time"].map(parse)
    df["_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    return df


def test_wash_adaylari_iterrows_referansiyla_ayni():
    df = _hazirla(_islemler_df(), _eski_parse_ts)
    beklenen = _eski_wash(df)
    assert beklenen, "kurgu dejenere: hiç wash adayı yok"
    seyrek = set(df.index[df["symbol"].isin(["SOLO", "DUO"])])
    assert not (seyrek & beklenen), "seyrek uç kural gereği aday OLMAMALI"
    assert tr._wash_adaylari(df) == beklenen


def test_lot_kaynak_sozlukleri_iterrows_ile_tip_ve_deger_denk():
    """`_lot_eslestir`in itertuples sözlükleri iterrows'unkiyle tip+değer olarak aynı (NaN dahil)."""
    df = _hazirla(_islemler_df(), _eski_parse_ts)
    wash = _eski_wash(df)
    eski = _eski_kaynak(df, wash)
    yeni = tr._kaynak_satirlari(df, wash)          # ÜRETİM yardımcısı (Codex: eskiden yeniden yazılıyordu)
    assert len(eski) == len(yeni)
    for a, b in zip(eski, yeni, strict=True):
        assert list(a) == list(b)
        for k in a:
            assert type(a[k]) is type(b[k]), k
            assert a[k] == b[k] or (pd.isna(a[k]) and pd.isna(b[k])), k


@pytest.mark.parametrize("lot_method", ["position", "fifo", "hifo"])
def test_build_eski_yardimcilarla_JSON_esit(tmp_path, monkeypatch, lot_method):
    p = tmp_path / "trades.csv"
    _islemler_df().to_csv(p, index=False)
    yeni = tr.build(p, lot_method=lot_method)
    monkeypatch.setattr(tr, "_parse_ts", _eski_parse_ts)
    monkeypatch.setattr(tr, "_wash_adaylari", _eski_wash)
    monkeypatch.setattr(tr, "_kaynak_satirlari", _eski_kaynak)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eski = tr.build(p, lot_method=lot_method)
    assert eski["n"] > 50 and eski["n_wash_candidates"] > 0
    assert json.dumps(yeni, sort_keys=True, default=repr) == json.dumps(eski, sort_keys=True, default=repr)


# ---------------------------------------------------------------------------
# tahmin defteri — kısa/uzun/boş satırlı KURGU
# ---------------------------------------------------------------------------
_BAR0 = datetime(2026, 9, 1, 14, 30)


def _defter(path: Path, legacy: bool = False) -> None:
    alanlar = [f for f in pt.FIELDS if f != "ai_side"] if legacy else pt.FIELDS
    rng = random.Random(3)
    satirlar: list[list[str]] = [alanlar]
    for i in range(160):
        sym = ["AAA", "BBB"][i % 2]
        ts = (_BAR0 + pd.Timedelta(minutes=i)).isoformat()
        if i % 7 == 0:
            ts = ts + "+00:00"                       # tz-aware kayıt (kanonik eşleşme yolu)
        d = {k: "" for k in alanlar}
        d.update(logged_at="2026-09-01T15:00:00", symbol=sym, interval="1m", bar_ts=ts,
                 direction=rng.choice("+-"), action="LONG", confidence="0.55", price=f"{100 + i * 0.01:.2f}",
                 bar_close=f"{100 + i * 0.01 - 0.005:.3f}", horizon_bars="5")
        if i < 40:
            d.update(resolved_at="2026-09-01T16:00:00", fwd_price="100.5", hit="1", fwd_ret_pct="0.1")
        satirlar.append([d[k] for k in alanlar])
    satirlar[60] = satirlar[60][:13]                  # KISA satır (resolved_at'tan itibaren yok) → çözülür
    satirlar[61] = satirlar[61] + ["fazla1", "fazla2"]  # UZUN satır → fazlalık yazımda düşer
    satirlar[10] = satirlar[10][:17]                  # çözülmüş bölgede kısa satır (yazımda '' ile dolar)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for j, s in enumerate(satirlar):
            w.writerow(s)
            if j in (20, 90):
                f.write("\r\n")                       # boş satır (DictReader atlar)


class _Saglayici:
    def get_bars(self, sym, iv, lookback_bars=400):
        idx = pd.date_range(_BAR0, periods=150, freq="1min")   # son 10 bar yok → bir kısmı pending
        return pd.DataFrame({"close": np.linspace(99.0, 101.0, len(idx))}, index=idx)


class _SabitZaman(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, 16, 12, 0, 0)


def test_dict_row_DictReader_ile_birebir(tmp_path):
    p = tmp_path / "log.csv"
    _defter(p)
    with p.open(encoding="utf-8", newline="") as f:
        beklenen = list(csv.DictReader(f))
    with p.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        h = next(r)
        yeni = [pt._dict_row(h, row) for row in r if row]
    assert yeni == beklenen
    assert any(None in d for d in beklenen) and any(d.get("ai_side") is None for d in beklenen)


def test_resolve_hizli_yeniden_yazim_eski_yolla_BAYT_DENK(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "datetime", _SabitZaman)
    sonuc = {}
    for ad in ("yeni", "eski"):
        p = tmp_path / f"{ad}.csv"
        _defter(p)
        monkeypatch.setattr(pt, "PRED_LOG", p)
        monkeypatch.setattr(pt, "_SUMMARY", None)
        if ad == "eski":
            monkeypatch.setattr(pt, "_rewrite_resolved_fast", lambda tmp, res: None)
        d = pt.resolve(_Saglayici())
        devir_var = pt._ROWS_HANDOFF is not None
        sonuc[ad] = (d, p.read_bytes(), devir_var, pt.summary())
    (d_yeni, b_yeni, devir_yeni, s_yeni), (d_eski, b_eski, devir_eski, s_eski) = sonuc["yeni"], sonuc["eski"]
    assert d_yeni == d_eski
    assert d_yeni["resolved"] > 50 and d_yeni["pending"] > 0          # kurgu dejenere değil
    assert d_yeni["checked"] == 160                                    # boş satırlar sayılmaz
    assert b_yeni == b_eski
    assert devir_yeni and not devir_eski                               # devir yalnız hızlı yolda
    assert json.dumps(s_yeni, sort_keys=True, default=repr) == json.dumps(s_eski, sort_keys=True, default=repr)
    assert pt._ROWS_HANDOFF is None                                    # tek seferlik
    kisa = tmp_path / "kontrol.csv"
    _defter(kisa)
    assert b_yeni != kisa.read_bytes()                                 # gerçekten yazıldı


def test_resolve_legacy_baslikta_hizli_yol_devreye_girmez(tmp_path, monkeypatch):
    p = tmp_path / "legacy.csv"
    _defter(p, legacy=True)
    monkeypatch.setattr(pt, "PRED_LOG", p)
    assert pt._rewrite_resolved_fast(tmp_path / "t.csv", {("AAA", "1m", "x"): {"hit": "1"}}) is None
    assert not (tmp_path / "t.csv").exists()


@pytest.mark.parametrize("kesik", [6, 12])
def test_yarim_yazilmis_son_satir_resolve_u_DUSURMEZ(tmp_path, monkeypatch, kesik):
    """BULGU-80-1: fiyatına (6) ya da ufkuna (12) kadar kesilmiş bekleyen tek satır eskiden
    TypeError/ValueError ile TÜM çözümü düşürüyordu; artık yalnız o satır bekliyor sayılır."""
    monkeypatch.setattr(pt, "datetime", _SabitZaman)
    p = tmp_path / "log.csv"
    _defter(p)
    monkeypatch.setattr(pt, "PRED_LOG", p)
    temiz = pt.resolve(_Saglayici())
    _defter(p)
    # Kesik satır ÇÖZÜLEBİLİR bir bara ait olmalı; yoksa `_forward_price` None verir ve fiyat
    # ayrıştırmasına hiç ulaşılmaz (ilk sürümde fiyat koruması mutasyonu bu yüzden KAÇTI).
    hedef = (_BAR0 + pd.Timedelta(minutes=50)).isoformat()
    with p.open(encoding="utf-8", newline="") as f:
        son = next(r for r in csv.reader(f) if len(r) > 3 and r[3] == hedef)
    son[1] = "YARIM"
    if kesik == 12:
        son[12] = "5x"                  # ufuk sayı değil
    with p.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(son[:kesik] if kesik == 6 else son)
    bozuk = pt.resolve(_Saglayici())
    assert bozuk["resolved"] == temiz["resolved"] > 0
    assert bozuk["pending"] == temiz["pending"] + 1 and bozuk["checked"] == temiz["checked"] + 1


def test_satir_devri_dosya_degisince_KULLANILMAZ(tmp_path, monkeypatch):
    """resolve() sonrası araya bir record() eklemesi girerse devir imzası tutmaz → dosya okunur."""
    monkeypatch.setattr(pt, "datetime", _SabitZaman)
    p = tmp_path / "log.csv"
    _defter(p)
    monkeypatch.setattr(pt, "PRED_LOG", p)
    pt.resolve(_Saglayici())
    assert pt._ROWS_HANDOFF is not None
    with p.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["2026-09-16T12:00:01", "ZZZ", "1m", "2026-09-16T12:00:00"] + [""] * 35)
    satirlar = pt._read_log_rows()
    with p.open(encoding="utf-8", newline="") as f:
        assert satirlar == list(csv.DictReader(f))
    assert satirlar[-1]["symbol"] == "ZZZ" and pt._ROWS_HANDOFF is None


def test_satir_devri_normal_yolda_KULLANILIR_ve_baska_surec_eklemesinde_REDDEDILIR(tmp_path, monkeypatch):
    """Devir imzası ara dosyadan değiştirmeden ÖNCE alınır. Değiştirmenin HEMEN ardından başka bir
    süreç satır eklerse (Codex incelemesi: imza sonradan alındığında eski satırlarla eşleşiyordu)
    devir reddedilmeli; hiçbir şey araya girmezse devir gerçekten kullanılmalı (ayrıştırma YOK)."""
    monkeypatch.setattr(pt, "datetime", _SabitZaman)
    p = tmp_path / "log.csv"
    monkeypatch.setattr(pt, "PRED_LOG", p)

    class _AyristirmaYasak:
        def __init__(self, *a, **k):
            raise AssertionError("devir varken DictReader çağrıldı")

    _defter(p)
    pt.resolve(_Saglayici())
    with monkeypatch.context() as m:
        m.setattr(pt.csv, "DictReader", _AyristirmaYasak)
        kullanilan = pt._read_log_rows()                      # yeniden adlandırma imzayı korumalı
    with p.open(encoding="utf-8", newline="") as f:
        assert kullanilan == list(csv.DictReader(f))

    _defter(p)
    gercek = pt._atomic_replace

    def degistir_ve_ekle(src, dst):
        gercek(src, dst)
        with open(dst, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["2026-09-16T12:00:01", "ZZZ", "1m", "2026-09-16T12:00:00"] + [""] * 35)

    monkeypatch.setattr(pt, "_atomic_replace", degistir_ve_ekle)
    pt.resolve(_Saglayici())
    satirlar = pt._read_log_rows()
    assert any(r["symbol"] == "ZZZ" for r in satirlar)


def test_satir_devri_arada_ozet_dolarsa_bellekte_KALMAZ(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "datetime", _SabitZaman)
    p = tmp_path / "log.csv"
    _defter(p)
    monkeypatch.setattr(pt, "PRED_LOG", p)
    monkeypatch.setattr(pt, "_SUMMARY", None)
    gercek = pt._atomic_replace

    def degistir_ve_ozetle(src, dst):
        gercek(src, dst)
        pt.summary()                                           # araya giren okuyucu önbelleği doldurur

    monkeypatch.setattr(pt, "_atomic_replace", degistir_ve_ozetle)
    pt.resolve(_Saglayici())
    assert pt._ROWS_HANDOFF is not None
    pt.summary()
    assert pt._ROWS_HANDOFF is None


def test_existing_keys_DictReader_referansiyla_ayni(tmp_path, monkeypatch):
    p = tmp_path / "log.csv"
    _defter(p)
    monkeypatch.setattr(pt, "PRED_LOG", p)
    with p.open(encoding="utf-8", newline="") as f:
        beklenen = {(r["symbol"], pt._canon_ts(r["bar_ts"])) for r in csv.DictReader(f)}
    assert pt._existing_keys() == beklenen
    # başlıkta ad yok: veri satırı VARSA None (DictReader KeyError), YOKSA boş küme
    p.write_text("a,b\r\n1,2\r\n", encoding="utf-8")
    assert pt._existing_keys() is None
    p.write_text("a,b\r\n", encoding="utf-8")
    assert pt._existing_keys() == set()


# ---------------------------------------------------------------------------
# options_analyzer — vollib (scipy.stats) tembel import
# ---------------------------------------------------------------------------
def test_options_analyzer_importu_scipy_stats_yuklemez(tmp_path):
    env = dict(os.environ, NIHAI_DATA_DIR=str(tmp_path / "d"), NIHAI_LOG_DIR=str(tmp_path / "l"),
               NIHAI_DISABLE_FILE_LOG="1")
    kod = ("import sys, nihai.risk.options_analyzer as oa; "
           "print('scipy.stats' in sys.modules, oa._VOLLIB_FNS is None)")
    out = subprocess.run([sys.executable, "-c", kod], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=120, check=True).stdout.split()
    assert out == ["False", "True"]


def test_gozlemevi_hmm_SABIT_pencerede_her_60_yeni_barda_yeniden_fit(monkeypatch):
    """BULGU-80-2: canlı servis SABİT 220 barlık pencere çeker; yeni bar sayısı uzunluktan
    (hep 0) değil zaman damgasından sayılmalı. Eski kural 300 yeni barda 1 fit veriyordu."""
    import nihai.ai.observatory as ob
    import nihai.ai.regime_hmm as rh

    fitler: list[int] = []
    gercek = rh.gaussian_hmm_fit
    monkeypatch.setattr(rh, "gaussian_hmm_fit", lambda x, **k: (fitler.append(len(x)), gercek(x, **k))[1])
    monkeypatch.setattr(ob, "_HMM_CACHE", {})
    monkeypatch.setenv("AI_OBS_HMM_REFIT_BARS", "60")
    rng = np.random.default_rng(0)
    fiyat = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    idx = pd.date_range("2026-09-01 14:30", periods=600, freq="1min")
    for t in range(220, 520):
        pencere = pd.Series(fiyat[t - 220:t], index=idx[t - 220:t])
        logret = np.diff(np.log(pencere.to_numpy()))
        ob._hmm_model("SYM", str(pencere.index[-1]), logret, pencere.index)
        ob._hmm_model("SYM", str(pencere.index[-1]), logret, pencere.index)   # aynı bar: fit YOK
    assert fitler == [219] * 5          # t=220 + her 60 yeni barda bir (280/340/400/460)


@pytest.mark.parametrize("refit,n_bar,tz_cevir,beklenen", [
    (300, 900, None, [0, 300, 600]),      # eşik pencereden (220) BÜYÜK → pencereyle sınırlanmamalı
    (60, 300, 250, [0, 60]),              # aynı anlar US/Eastern GÖSTERİMİYLE gelince erken fit YOK
])
def test_gozlemevi_hmm_sayaci_pencereden_ve_gosterimden_bagimsiz(monkeypatch, refit, n_bar, tz_cevir, beklenen):
    """Codex incelemesi iki kusuru ölçtü: damga arama pencereyle sınırlıydı (refit=300 → 220'de bir)
    ve `str()` gösterimi karşılaştırıldığı için saat dilimi değişince yeni bar gelmeden fit ediliyordu."""
    import nihai.ai.observatory as ob
    import nihai.ai.regime_hmm as rh

    adimlar: list[int] = []
    gercek = rh.gaussian_hmm_fit
    monkeypatch.setattr(ob, "_HMM_CACHE", {})
    monkeypatch.setenv("AI_OBS_HMM_REFIT_BARS", str(refit))
    rng = np.random.default_rng(0)
    fiyat = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bar)))
    idx = pd.date_range("2026-09-01 14:30", periods=n_bar, freq="1min", tz="UTC")
    t_simdi = {"t": 0}
    monkeypatch.setattr(rh, "gaussian_hmm_fit",
                        lambda x, **k: (adimlar.append(t_simdi["t"] - 220), gercek(x, **k))[1])
    for t in range(220, n_bar):
        t_simdi["t"] = t
        ix = idx[t - 220:t]
        if tz_cevir is not None and t >= tz_cevir:
            ix = ix.tz_convert("US/Eastern")
        logret = np.diff(np.log(fiyat[t - 220:t]))
        ob._hmm_model("SYM", str(ix[-1]), logret, ix)
        if tz_cevir is not None and t == tz_cevir:      # aynı bar, farklı gösterim — fit YOK
            ob._hmm_model("SYM", str(ix[-1]), logret, ix)
    assert adimlar == beklenen


def test_gozlemevi_hmm_bar_araligi_basina_AYRI_model(monkeypatch):
    """BULGU-80-4: aynı sembol tahmin servisinden 1m, tarama köprüsünden 1h barla gelir; anahtar
    `sembol|durum` iken ilk fit edilen aralığın modeli ötekinin rejim olasılığında kullanılıyordu."""
    import nihai.ai.observatory as ob
    import nihai.ai.regime_hmm as rh

    fitler: list[int] = []
    gercek = rh.gaussian_hmm_fit
    monkeypatch.setattr(rh, "gaussian_hmm_fit", lambda x, **k: (fitler.append(len(x)), gercek(x, **k))[1])
    monkeypatch.setattr(ob, "_HMM_CACHE", {})
    rng = np.random.default_rng(3)
    dk = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 220))),
                   index=pd.date_range("2026-09-01 14:30", periods=220, freq="1min"))
    sa = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
                   index=pd.date_range("2026-06-01 14:30", periods=200, freq="1h"))
    m_dk = ob._hmm_model("SYM", str(dk.index[-1]), np.diff(np.log(dk.to_numpy())), dk.index)
    m_sa = ob._hmm_model("SYM", str(sa.index[-1]), np.diff(np.log(sa.to_numpy())), sa.index)
    assert fitler == [219, 199]
    assert m_dk is not m_sa and len(ob._HMM_CACHE) == 2


def test_mc_dropout_istisnada_model_eval_kipine_doner():
    _neural, m, X = _kucuk_model()
    gercek = m._model.forward
    sayac = {"n": 0}

    def patlayan(*a, **k):
        sayac["n"] += 1
        if sayac["n"] == 2:
            raise RuntimeError("cihaz hatası")
        return gercek(*a, **k)

    m._model.forward = patlayan
    with pytest.raises(RuntimeError):
        m.predict_with_uncertainty(X[:1], T=5)
    assert m._model.training is False


def _kucuk_model():
    from nihai.ai import neural
    from nihai.signals.meta_features import FEATURE_NAMES
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(float)
    return neural, neural.NeuralCapacity("deep_mlp", feature_names=list(FEATURE_NAMES), seed=3).fit(
        X, y, max_epochs=1), X


def test_mc_dropout_ES_ZAMANLI_cagrida_seri_sonuçla_birebir():
    """BULGU-80-3: süreç-genel tohum + paylaşılan train/eval kipi → kilitsiz 8 thread'de
    sonuçların %97'si seriden farklıydı (gerçek artefakt). Kilitle birebir aynı olmalı."""
    import threading

    _neural, m, X = _kucuk_model()
    satirlar = [X[i:i + 1] for i in range(30)]
    seri = [m.predict_with_uncertainty(r, T=10) for r in satirlar]
    sonuc: dict[int, list] = {}

    def is_(k: int) -> None:
        sonuc[k] = [(i, m.predict_with_uncertainty(satirlar[i], T=10))
                    for _ in range(3) for i in range(len(satirlar))]

    ts = [threading.Thread(target=is_, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    fark = sum(1 for out in sonuc.values() for i, (p, sd) in out
               if not (np.array_equal(p, seri[i][0]) and np.array_equal(sd, seri[i][1])))
    assert sum(len(o) for o in sonuc.values()) == 8 * 3 * 30
    assert fark == 0


def test_mc_dropout_sirasinda_ES_ZAMANLI_model_yuklemesi_sonucu_BOZMAZ(tmp_path):
    """Codex incelemesi: `load()` → `_build_torch_model` kilitsiz `torch.manual_seed` çağırıp RNG
    tüketiyordu; MC-dropout'un ilk geçişinden sonra araya girince sonuç değişiyordu (doğrulandı)."""
    import threading

    neural, m, X = _kucuk_model()
    base = tmp_path / "model"
    m.save(base)
    referans = m.predict_with_uncertainty(X[:1], T=10)
    ilk_gecis, yukleme_bitti = threading.Event(), threading.Event()
    gercek_forward = m._model.forward
    sayac = {"n": 0}

    def bekleyen_forward(*a, **k):
        sayac["n"] += 1
        if sayac["n"] == 1:
            ilk_gecis.set()
            yukleme_bitti.wait(timeout=0.5)   # kilit varken yükleme BEKLER → zaman aşımıyla devam
        return gercek_forward(*a, **k)

    m._model.forward = bekleyen_forward

    def yukle():
        ilk_gecis.wait(timeout=5.0)
        neural.NeuralCapacity.load(base)
        yukleme_bitti.set()

    th = threading.Thread(target=yukle)
    th.start()
    sonuc = m.predict_with_uncertainty(X[:1], T=10)
    th.join(timeout=30)
    assert yukleme_bitti.is_set()
    assert np.array_equal(sonuc[0], referans[0]) and np.array_equal(sonuc[1], referans[1])


def test_has_vollib_bulunan_ama_import_EDILEMEYEN_pakette_False(tmp_path):
    """Codex incelemesi: tembel sürümde bayrak ilk Greek çağrısına kadar True kalıyordu."""
    sahte = tmp_path / "sahte" / "vollib"
    (sahte / "black_scholes").mkdir(parents=True)
    (sahte / "__init__.py").write_text("", encoding="utf-8")
    (sahte / "black_scholes" / "__init__.py").write_text("raise ImportError('bozuk kurulum')\n", encoding="utf-8")
    # Tur-81 (BULGU-81-4): alt süreç `-c` ile koşuyor, yani bir `__main__` GİRİŞ NOKTASI
    # DEĞİL → `console.ensure_utf8_stdout()` hiç çağrılmıyor ve loguru'nun stderr'i
    # Windows konsol kod sayfasına (cp1254) düşüyor. Ebeveyn onu utf-8 + `errors="replace"`
    # ile okuyunca "bulunamadı"nın `ı`sı U+FFFD oluyor ve iddia ORTAMA göre kırılıyordu
    # (üretim yolu etkilenmez: 74 giriş noktası reconfigure'ü çağırır, testle kilitli).
    # `PYTHONIOENCODING` alt sürece giriş noktasının kurduğu ortamın AYNISINI verir.
    env = dict(os.environ, NIHAI_DATA_DIR=str(tmp_path / "d"), NIHAI_LOG_DIR=str(tmp_path / "l"),
               NIHAI_DISABLE_FILE_LOG="1", PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([str(tmp_path / "sahte"), str(ROOT)]))
    kod = "import nihai.risk.options_analyzer as oa; print(oa.HAS_VOLLIB)"
    r = subprocess.run([sys.executable, "-c", kod], cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120, check=True)
    assert r.stdout.split() == ["False"]
    assert "vollib bulunamadı" in r.stderr


def test_inference_threads_kapsamli_ve_hata_da_geri_alinir():
    neural, _m, _X = _kucuk_model()
    import torch

    once = torch.get_num_threads()
    with neural.inference_threads(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == once
    with pytest.raises(RuntimeError), neural.inference_threads(1):
        raise RuntimeError("çıkarım hatası")
    assert torch.get_num_threads() == once


def test_golge_pwin_tek_ic_thread_ve_kilit_altinda(monkeypatch):
    from nihai.ai import neural
    from nihai.execution import neural_shadow as ns
    from nihai.signals.meta_features import FEATURE_NAMES
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")
    import torch

    gorulen: dict = {}

    class _Model:
        tier = "deep_mlp"
        feature_names = list(FEATURE_NAMES)

        def predict_with_uncertainty(self, X, T=20):
            gorulen["threads"] = torch.get_num_threads()
            gorulen["kilit"] = neural.INFER_LOCK._is_owned()
            return [0.6], [0.05]

    sh = ns._NeuralShadow.__new__(ns._NeuralShadow)
    sh.tier, sh.card_hash, sh.reason, sh._passes = "deep_mlp", "x", None, 5
    sh._model, sh._features, sh._max_z = _Model(), list(FEATURE_NAMES), 0
    assert sh.pwin({n: 0.0 for n in FEATURE_NAMES}, "LONG") == (0.6, 0.05)
    assert gorulen == {"threads": 1, "kilit": True}


def test_compute_greeks_tembel_vollib_dalini_kullanir():
    from nihai.risk import options_analyzer as oa
    if not oa.HAS_VOLLIB:
        pytest.skip("vollib kurulu değil")
    g = oa.compute_greeks(option_type="c", S=100.0, K=105.0, T=0.25, r=0.05, sigma=0.3)
    assert g is not None and oa._VOLLIB_FNS is not None
    assert g.delta == pytest.approx(oa._VOLLIB_FNS[0]("c", 100.0, 105.0, 0.25, 0.05, 0.3), abs=0)
