"""Tur-102 — gölge/danışman 5 mekanizmaya ayrı sanal kasa (`nihai/execution/golge_kasalar.py`).

Dört soru kilitlenir:
  1. KURAL: her kasa mekanizmasının KENDİ enforce sözleşmesini mi uyguluyor (saf `karar`)?
  2. MOTOR: backtest sözleşmesi (karar T kapanışı → giriş T+1 açılışı, önce çıkış sonra giriş), veto,
     sembol başına tek pozisyon, bayat emir iptali, yalnız KAPANMIŞ bar, yeniden başlatmada süreklilik.
  3. YALITIM: canlı portföye/üretim defterlerine dokunmuyor, hata tahmin servisini düşürmüyor.
  4. BAĞLANTI: tahmin servisi çağırıyor (seans açıkken), API + pano okuyor, ayarlar beyanlı.
"""
from __future__ import annotations

import csv
import inspect
from pathlib import Path

import pandas as pd
import pytest

from nihai.execution import ai_advisor, shadow_bridge
from nihai.execution import golge_kasalar as G
from nihai.risk.prob_sizing import prob_to_size_mult
from nihai.signals.strategy import Signal, SignalAction

KOK = Path(__file__).resolve().parents[1]
SEMBOL = "AAA"


# --------------------------------------------------------------------------- yardımcılar
def _barlar(n: int, bas: str = "2026-09-28 14:00", fiyat: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(bas, periods=n, freq="1min")
    return pd.DataFrame({"open": fiyat, "high": fiyat + 0.1, "low": fiyat - 0.1, "close": fiyat,
                         "volume": 1000.0}, index=idx)


def _sinyal(fiyat: float = 100.0, taraf: str = "LONG") -> Signal:
    stop, hedef = (fiyat - 1.0, fiyat + 3.0) if taraf == "LONG" else (fiyat + 1.0, fiyat - 3.0)
    return Signal(SignalAction(taraf), SEMBOL, fiyat, stop, hedef, reasons=["test"],
                  metadata={"strategy": "EventBreakout"})


def _kur(monkeypatch, tmp_path, *, aday_barlari: set, olcum: G.Olcum, taraf: str = "LONG") -> G.Motor:
    """Adayı belirli barlarda üreten, ölçümü sabit veren motor (tek değişken: kasa kuralı)."""
    def aday(symbol, df, shared=None):
        return _sinyal(float(df["close"].iloc[-1]), taraf) if df.index[-1] in aday_barlari else None

    monkeypatch.setattr(G, "aday", aday)
    monkeypatch.setattr(G, "olc", lambda symbol, df, sig, shared=None, *, conf_bar=None: olcum)
    return G.Motor(kok=tmp_path, bakiye=1_000_000.0)


def _yuru(m: G.Motor, df: pd.DataFrame, bas: int = 0) -> None:
    """Barları canlıdaki gibi tek tek besle: her çağrıda son bar yeni kapanmıştır."""
    for k in range(bas, len(df)):
        m.bar_isle(SEMBOL, df.iloc[: k + 1],
                   simdi=(df.index[k] + pd.Timedelta(seconds=61)).to_pydatetime())


def _kararlar(kok: Path) -> list[dict]:
    with (kok / "kararlar.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _islemler(kok: Path) -> list[dict]:
    yol = kok / "islemler.csv"
    if not yol.exists():
        return []
    with yol.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- 1. KURAL (saf)
def _k(kasa, **alan):
    ne = alan.pop("_ne", 0.486)
    return G.karar(kasa, G.Olcum(**alan), meta_esik=0.45, noral_esik=ne)


def test_kasalar_ve_varsayilan_bakiye(tmp_path):
    assert G.KASALAR == ("kontrol", "meta_kapi", "ai_karar", "golge_kopru", "noral", "olasilik_boyut")
    m = G.Motor(kok=tmp_path)
    for ad in G.KASALAR:
        assert m.kasalar[ad].bakiye == 1_000_000.0
        assert m.kasalar[ad].portfoy.equity == 1_000_000.0


def test_kontrol_her_adayi_tam_riskle_alir():
    assert _k("kontrol", meta_p=0.01, ai_veto=1, karantina=True) == (True, 1.0, "taban")


def test_meta_kapi_strategy_enforce_dali():
    assert _k("meta_kapi", meta_p=0.44)[0] is False
    assert _k("meta_kapi", meta_p=0.45)[:2] == (True, 1.0)
    assert _k("meta_kapi")[:2] == (True, 1.0)            # p yok → geçer (strategy.py enforce ile aynı)


def test_ai_karar_advise_enforce_dali():
    # Tur-105: ABSTAIN artık veto DEĞİL (ayrıntı tests/test_tur105_ai_kasa_abstain.py); ters yön veto
    assert _k("ai_karar", ai_action="SHORT_BIAS", ai_post=0.39, ai_veto=1)[0] is False
    gir, carpan, _ = _k("ai_karar", ai_action="LONG_BIAS", ai_post=0.55, ai_veto=0)
    assert gir and carpan == pytest.approx(ai_advisor.enforce_boyut(0.55)) == pytest.approx(0.625)
    assert _k("ai_karar")[:2] == (True, 1.0)              # verdikt yok → advise() None gibi, veto yok


def test_golge_kopru_dondurulmus_kurallar_ve_karantina():
    assert _k("golge_kopru", karantina=True, conf_bar=0.9, value_area_pos=1.0)[0] is False
    assert _k("golge_kopru", karantina=False, conf_bar=0.59, value_area_pos=1.0)[0] is False
    assert _k("golge_kopru", karantina=False, conf_bar=0.60, value_area_pos=0.0)[0] is False
    assert _k("golge_kopru", karantina=False, conf_bar=0.60, value_area_pos=1.0)[:2] == (True, 1.0)
    assert _k("golge_kopru")[:2] == (True, 1.0)           # ölçüm yok → geçer, gerekçede yazılı
    assert "ölçüm yok" in _k("golge_kopru")[2]


def test_noral_ısınmada_ve_olcumsuz_girmez():
    assert _k("noral", noral_p=0.49, _ne=None)[0] is False
    assert _k("noral", _ne=0.486)[0] is False             # model görüş bildirmedi → girmez
    assert _k("noral", noral_p=0.4859, _ne=0.486)[0] is False
    assert _k("noral", noral_p=0.486, _ne=0.486)[:2] == (True, 1.0)


def test_olasilik_boyut_veto_etmez_boyutu_kucultur():
    gir, carpan, _ = _k("olasilik_boyut", meta_p=0.2, guven=0.5)
    assert gir and carpan == pytest.approx(0.5 * prob_to_size_mult(0.2, p_lo=0.45, p_hi=0.60, floor=0.25))
    assert _k("olasilik_boyut", meta_p=0.9, guven=1.0)[1] == 1.0


def test_bilinmeyen_kasa_hata_verir():
    with pytest.raises(ValueError):
        _k("yok")


# --------------------------------------------------------------------------- 2. MOTOR
def test_giris_SONRAKI_barin_acilisinda_ve_vetolar_uygulanir(monkeypatch, tmp_path):
    df = _barlar(6)
    df.iloc[3, df.columns.get_loc("open")] = 100.5          # karar barı 2 → giriş barı 3'ün açılışı (kapanış 100:
    df.iloc[3, df.columns.get_loc("high")] = 100.6          #  %0,5 ayrık — kapanıştan dolum ayırt edilsin, Tur-102 M1;
    #  Tur-106: %1'lik kovalama RR'ı 3 → 1,0'a indirip canlı RR kapısına takılırdı, dolumda RR 1,5 kalsın)
    m = _kur(monkeypatch, tmp_path, aday_barlari={df.index[2]},
             olcum=G.Olcum(meta_p=0.2, noral_p=0.49, ai_action="ABSTAIN", ai_post=0.39, ai_veto=1,
                           karantina=False, guven=0.5))
    _yuru(m, df.iloc[:3])
    assert all(not k.portfoy.open_positions for k in m.kasalar.values())   # karar barında pozisyon YOK
    _yuru(m, df, bas=3)
    acik = {ad: k.portfoy.open_positions for ad, k in m.kasalar.items()}
    assert len(acik["kontrol"]) == 1 and abs(acik["kontrol"][0].entry_price / 100.5 - 1) < 0.003
    assert acik["meta_kapi"] == [] and acik["noral"] == []                             # veto + ısınma
    assert len(acik["golge_kopru"]) == 1 and len(acik["olasilik_boyut"]) == 1
    assert acik["olasilik_boyut"][0].quantity < acik["kontrol"][0].quantity             # boyut küçüldü
    assert len(acik["ai_karar"]) == 1                                                   # Tur-105: ABSTAIN girer
    assert acik["ai_karar"][0].quantity < acik["kontrol"][0].quantity                   # post 0,39 → ×0,25
    satir = _kararlar(tmp_path)[0]
    assert satir["kontrol"].startswith("gir") and satir["meta_kapi"].startswith("veto")


def test_acik_pozisyonda_yeni_aday_dolu_ve_stop_cikisi_defterde(monkeypatch, tmp_path):
    df = _barlar(8)
    df.iloc[6, df.columns.get_loc("low")] = 98.0               # stop 99 delinir (açılış 100: GAP değil)
    m = _kur(monkeypatch, tmp_path, aday_barlari={df.index[1], df.index[4]}, olcum=G.Olcum())
    _yuru(m, df)
    assert _kararlar(tmp_path)[1]["kontrol"] == "dolu"
    kontrol = [r for r in _islemler(tmp_path) if r["kasa"] == "kontrol"]
    assert len(kontrol) == 1 and "STOP" in kontrol[0]["tetik"] and float(kontrol[0]["kz"]) < 0
    assert m.kasalar["kontrol"].portfoy.equity < 1_000_000.0
    assert m.kasalar["kontrol"].portfoy.equity == pytest.approx(float(kontrol[0]["ozsermaye"]))


def test_bayat_emir_iptal_edilir(monkeypatch, tmp_path):
    df = pd.concat([_barlar(3), _barlar(2, bas="2026-09-28 14:20")])   # 18 dk boşluk (seans sonu benzeri)
    m = _kur(monkeypatch, tmp_path, aday_barlari={df.index[2]}, olcum=G.Olcum())
    _yuru(m, df)
    assert all(not k.portfoy.open_positions for k in m.kasalar.values())
    assert m.sayac["iptal"] >= 1


def test_yalniz_KAPANMIS_bar_islenir():
    df = _barlar(3)
    son = df.index[-1]
    assert len(G.kapali_barlar(df, (son + pd.Timedelta(seconds=30)).to_pydatetime())) == 2
    assert len(G.kapali_barlar(df, (son + pd.Timedelta(seconds=60)).to_pydatetime())) == 3
    tz = df.tz_localize("UTC").tz_convert("America/New_York")           # tz-farkında indeks aynı sonucu verir
    assert len(G.kapali_barlar(tz, (son + pd.Timedelta(seconds=30)).to_pydatetime())) == 2


def test_yeniden_baslatmada_pozisyon_ve_kacan_barlar_surer(monkeypatch, tmp_path):
    df = _barlar(9)
    df.iloc[7, df.columns.get_loc("low")] = 98.0
    m = _kur(monkeypatch, tmp_path, aday_barlari={df.index[1]}, olcum=G.Olcum())
    _yuru(m, df.iloc[:4])
    assert len(m.kasalar["kontrol"].portfoy.open_positions) == 1
    yeni = G.Motor(kok=tmp_path, bakiye=1_000_000.0)                     # süreç yeniden başladı
    assert len(yeni.kasalar["kontrol"].portfoy.open_positions) == 1
    kayit = pd.Timestamp(yeni._son_bar[SEMBOL])                          # son TUTARLI anlık görüntü
    yeni.bar_isle(SEMBOL, df, simdi=(df.index[-1] + pd.Timedelta(seconds=61)).to_pydatetime())  # kaçan barlar birden
    assert yeni.sayac["bar"] == int((df.index > kayit).sum()) >= 5      # kayıttan sonraki HER bar, bir kez
    assert [r["kasa"] for r in _islemler(tmp_path)].count("kontrol") == 1


def test_noral_esigi_yalniz_ONCEKI_olcumlerin_medyani(monkeypatch, tmp_path):
    m = G.Motor(kok=tmp_path)
    ts = pd.Timestamp("2026-09-28 14:00")
    # 30 ölçüm: yarısı 0,40 yarısı 0,42 → önceki medyan 0,41. 0,405 KENDİ eşiğine girseydi medyan 0,405 olur ve
    # aday geçerdi — "yalnız önceki ölçümler" iddiasını ayırt eden kurgu budur (Tur-73: dejenere kurgu kaçırır).
    for i in range(G.NORAL_ISINMA + 2):
        p = (0.40, 0.42)[i % 2] if i < G.NORAL_ISINMA else (0.405 if i == G.NORAL_ISINMA else 0.43)
        m._aday_kaydet(f"S{i}", ts, _sinyal(), G.Olcum(noral_p=p))
    kr = _kararlar(tmp_path)
    assert all(r["noral"].startswith("veto: ısınma") for r in kr[: G.NORAL_ISINMA])
    assert kr[G.NORAL_ISINMA]["noral"].startswith("veto") and float(kr[G.NORAL_ISINMA]["noral_esik"]) == 0.41
    assert kr[G.NORAL_ISINMA + 1]["noral"].startswith("gir")
    assert list(G.Motor(kok=tmp_path)._noral)[-1] == 0.43                # pencere defterden yeniden kurulur


def test_aday_meta_enforce_elenen_yonu_geri_alir(monkeypatch):
    elenen = Signal(SignalAction.WAIT, SEMBOL, 100.0, 99.0, 103.0, metadata={"meta_gated": "SHORT"})
    monkeypatch.setattr(G, "generate_signal", lambda *a, **k: elenen)
    sig = G.aday(SEMBOL, _barlar(80))
    assert sig is not None and sig.action == SignalAction.SHORT and sig.stop == 99.0
    monkeypatch.setattr(G, "generate_signal", lambda *a, **k: Signal(SignalAction.WAIT, SEMBOL, 0, 0, 0))
    assert G.aday(SEMBOL, _barlar(80)) is None


# --------------------------------------------------------------------------- 3. YALITIM
def test_uretim_defterlerine_yazmaz(tmp_path):
    m = G.Motor(kok=tmp_path)
    assert all(k.pm.record_logs is False for k in m.kasalar.values())
    assert {k.store.path.parent for k in m.kasalar.values()} == {tmp_path}


def test_hata_tahmin_servisini_dusurmez_ve_gorunur(monkeypatch):
    class Bozuk:
        def bar_isle(self, *a, **k):
            raise RuntimeError("patladı")

    monkeypatch.setenv("GOLGE_KASA", "on")
    monkeypatch.setattr(G, "motor", lambda: Bozuk())
    monkeypatch.setattr(G, "_son_hata", None)
    G.isle(SEMBOL, _barlar(3))
    assert "bar_isle" in (G._son_hata or "") and "patladı" in G._son_hata


def test_kapaliyken_motor_hic_kurulmaz(monkeypatch):
    monkeypatch.setenv("GOLGE_KASA", "off")
    monkeypatch.setattr(G, "motor", lambda: pytest.fail("kapalıyken motor kuruldu"))
    G.isle(SEMBOL, _barlar(3))
    assert G.etkin() is False


def test_diskten_durum_yazmaz(tmp_path):
    d = G.durum_diskten(tmp_path)
    assert set(d["kasalar"]) == set(G.KASALAR) and list(tmp_path.iterdir()) == []
    assert all(k["mtm"] == 1_000_000.0 and k["fark_kontrol_pp"] == 0.0 for k in d["kasalar"].values())


# --------------------------------------------------------------------------- 4. BAĞLANTI + regresyon kapıları
def test_tahmin_servisi_seans_acikken_kasayi_cagirir():
    from nihai.execution.live_predict_service import PredictionService
    kaynak = inspect.getsource(PredictionService._bar_context)
    assert "golge_kasalar.isle(" in kaynak and "self._cycle_market_open is True" in kaynak
    assert kaynak.index("self._cycle_market_open is True") < kaynak.index("golge_kasalar.isle(")
    satir = inspect.getsource(PredictionService._row)
    assert "def _yon_guven(px: float)" not in satir and "self._yon_guven(ctx" in satir   # TEK kaynak


def test_testler_kasayi_varsayilan_kapatir():
    assert 'os.environ["GOLGE_KASA"] = "off"' in (KOK / "conftest.py").read_text(encoding="utf-8")


def test_ai_advise_boyutu_tek_kaynaktan():
    kaynak = inspect.getsource(ai_advisor.advise)
    assert "enforce_boyut(d.posterior)" in kaynak and "prob_to_size_mult(" not in kaynak


def test_kopru_kurallari_dondurulmus_ve_defter_bos():
    assert G.KOPRU_KURALLARI == {"conf_bar": 0.60, "value_area_pos": 1.0}   # Tur-88 ön-kaydı (BULGU-88-5)
    assert shadow_bridge.PROMOTED == (), "terfi defteri değişti → golge_kopru kasasının kuralını güncelle"


def test_api_ucu_kasalari_dondurur(monkeypatch, tmp_path):
    from nihai.web import api_extra
    monkeypatch.setenv("GOLGE_KASA_DIR", str(tmp_path))
    monkeypatch.setattr(G, "_MOTOR", None)
    d = api_extra.golge_kasalar_durum()
    assert d["ok"] is True and list(d["kasalar"]) == list(G.KASALAR)
    assert d["kasalar"]["kontrol"]["bakiye"] == 1_000_000.0


def test_pano_sekmesi_ve_modul_bagli():
    html = (KOK / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    assert "aria-controls=\"tab-kasalar\"" in html and 'id="tab-kasalar"' in html
    assert 'apiCall("golge_kasalar")' in html and 'id="golge-kasa-body"' in html
    assert '"m10-golge-kasalar"' in html


def test_ayarlar_beyanli():
    ornek = (KOK / ".env.example").read_text(encoding="utf-8")
    for ad in ("GOLGE_KASA=", "GOLGE_KASA_STRATEJI=", "GOLGE_KASA_BAKIYE=", "GOLGE_KASA_DIR="):
        assert ad in ornek, ad


# --------------------------------------------------------------------------- 5. tek yazar + ayrı koşucu
def test_kasa_dizinine_TEK_surec_yazar(monkeypatch, tmp_path):
    import subprocess
    import sys
    import time

    kod = ("import sys,time; from pathlib import Path; from nihai.execution import golge_kasalar as G; "
           "assert G._yazar_kilidi(Path(sys.argv[1])); print('TUTUYOR', flush=True); time.sleep(30)")
    cocuk = subprocess.Popen([sys.executable, "-c", kod, str(tmp_path)], cwd=KOK, stdout=subprocess.PIPE,
                             text=True, env={**__import__("os").environ, "PYTHONPATH": str(KOK)})
    try:
        assert cocuk.stdout is not None and cocuk.stdout.readline().strip() == "TUTUYOR"
        monkeypatch.setenv("GOLGE_KASA_DIR", str(tmp_path))
        monkeypatch.setattr(G, "_MOTOR", None)
        monkeypatch.setattr(G, "_SON_DENEME", 0.0)
        assert G.motor() is None                                   # başka süreç tutuyor → bu süreç dokunmaz
        monkeypatch.setenv("GOLGE_KASA", "on")
        G.isle(SEMBOL, _barlar(3))                                  # sessizce atlar, hata yok
        assert not (tmp_path / "kontrol.json").exists()
    finally:
        cocuk.kill()
        cocuk.wait()
    time.sleep(0.2)
    monkeypatch.setattr(G, "_SON_DENEME", 0.0)
    monkeypatch.setattr(G, "_YAZAR", None)
    m = G.motor()                                                   # kilit serbest → bu süreç alır
    assert m is not None and m.kok == tmp_path


def test_ayri_kosucu_tahmin_defterine_yazmaz(monkeypatch, tmp_path):
    from nihai.execution import live_predict_service as L

    gorulen = {}

    class Sahte:
        def start(self):
            gorulen["track"] = __import__("os").environ["PREDICT_TRACK"]

    monkeypatch.setenv("GOLGE_KASA_DIR", str(tmp_path))
    monkeypatch.setenv("PREDICT_TRACK", "true")
    monkeypatch.setenv("PREDICT_REFRESH_SEC", "5")
    monkeypatch.setenv("GOLGE_KASA", "off")          # kos() "on" yazar; test sonunda geri alınsın (sızmasın)
    monkeypatch.setattr(G, "_MOTOR", G.Motor(kok=tmp_path))
    monkeypatch.setattr(L, "get_service", lambda: Sahte())
    monkeypatch.setattr(G.time, "sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        G.kos()
    assert gorulen["track"] == "false" and __import__("os").environ["PREDICT_REFRESH_SEC"] == "15"


def test_m_ile_kosucu_KANONIK_modulde_calisir(monkeypatch, tmp_path):
    """`python -m` dosyayı `__main__` olarak yükler; kanca ise `nihai.execution.golge_kasalar`ı görür. Koşucu `__main__`
    kopyasında kalsaydı aynı süreçte iki motor olur, kanca kilidi "başka süreç" sanıp kasalara hiç dokunmazdı (ilk canlı
    denemede tam böyle oldu)."""
    import runpy
    import sys
    import warnings

    from nihai.execution import live_predict_service as L

    cagri: list[str] = []
    monkeypatch.setattr(G, "kos", lambda: cagri.append("kanonik"))
    monkeypatch.setattr(L, "get_service", lambda: pytest.fail("__main__ kopyası kendi kos()'unu çalıştırdı"))
    monkeypatch.setattr(sys, "argv", ["golge_kasalar", "--kos"])
    for ad, v in (("GOLGE_KASA_DIR", str(tmp_path)), ("GOLGE_KASA", "off"), ("PREDICT_TRACK", "true"),
                  ("PREDICT_REFRESH_SEC", "5")):
        monkeypatch.setenv(ad, v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # runpy: modül sys.modules'ta zaten var (beklenen)
        runpy.run_module("nihai.execution.golge_kasalar", run_name="__main__")
    assert cagri == ["kanonik"]


# --------------------------------------------------------------------------- 6. boyut (BULGU-102-7)
def test_kasa_adedi_pay_tavani_ve_carpan_SONRA():
    # canlı IBM örneği: stop 0,31 $ → risk kuralı 48.387 hisse; pay tavanı %10 → 455; çarpan 0,25 tavandan SONRA
    assert G.kasa_adedi(48_387, 219.45, 1_000_000.0, 1.0, 0.10) == 455
    assert G.kasa_adedi(48_387, 219.45, 1_000_000.0, 0.25, 0.10) == 113          # tavan bağlarken de küçülür
    assert G.kasa_adedi(100, 219.45, 1_000_000.0, 0.5, 0.10) == 50               # risk bağlıyorsa risk adedi
    assert G.kasa_adedi(100, 0.0, 1_000_000.0, 1.0, 0.10) == 0


def test_dar_stopta_carpan_etkili_ve_sermaye_yayiliyor(monkeypatch, tmp_path):
    df = _barlar(6)
    m = _kur(monkeypatch, tmp_path, aday_barlari={df.index[2]}, olcum=G.Olcum(meta_p=0.2, guven=1.0))
    monkeypatch.setattr(G, "aday", lambda symbol, df_, shared=None: (
        Signal(SignalAction.LONG, SEMBOL, 100.0, 99.85, 103.0, metadata={"strategy": "EventBreakout"})
        if df_.index[-1] == df.index[2] else None))                           # stop %0,15 → risk kuralı tavana çarpar
    _yuru(m, df)
    kontrol = m.kasalar["kontrol"].portfoy.open_positions[0]
    olasilik = m.kasalar["olasilik_boyut"].portfoy.open_positions[0]
    assert kontrol.quantity * kontrol.entry_price <= 0.10 * 1_000_000.0 * 1.01   # tek pozisyon ≤ özsermayenin %10'u
    assert olasilik.quantity == pytest.approx(kontrol.quantity * 0.25, abs=1)   # prob_to_size_mult(0,2) = 0,25
