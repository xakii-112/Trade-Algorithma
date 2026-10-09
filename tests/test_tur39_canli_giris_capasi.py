"""
test_tur39_canli_giris_capasi.py — Tur-39: sinyal fiyatı ≠ İŞLEM fiyatı.

ÖLÇÜLEN KUSUR (510 sanal işlem, 2026-07-06→08-20): `generate_signal` fiyatı
`df_htf["close"].iloc[-1]`'dir ve `HTF_INTERVAL=1h` iken ~59 dk BAYAT olabilir; pozisyon
ise 1m barlarla/kotasyonla yönetilir. Sonuç:
  · işlemlerin **%46'sında MFE tam 0,00** — girilen fiyat piyasada YOKTU,
  · `GAP_STOP` sınıfı **68/68 aleyhte**, ort. −%2,45 / −1,497R, toplam **−299.438 $**
    (sistemin TÜM net zararından büyük; bu sınıf olmadan kalan 442 işlem +41.804 $),
  · 27'si girişten sonra **2 dakikadan kısa sürede** tetiklenmiş.
Gerçek bir gap dağılımı simetriktir; 68/68 aleyhte olmak piyasa değil ÖLÇÜM kusurudur.

Bu dosya onarımı kilitler: giriş canlı fiyata çapalanır, stop/target YAPISAL kalır, RR
yeniden hesaplanır ve kurulum artık geçerli değilse İŞLEM AÇILMAZ.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nihai.signals.strategy import Signal, SignalAction, reanchor_to_live

_MAIN = Path(__file__).resolve().parents[1] / "nihai" / "execution" / "main.py"


def _long(price=100.0, stop=98.0, target=104.0) -> Signal:
    return Signal(action=SignalAction.LONG, symbol="AAA", price=price, stop=stop,
                  target=target, reasons=["test"], quality_score=3.0, rr_ratio=2.0)


def _short(price=100.0, stop=102.0, target=96.0) -> Signal:
    return Signal(action=SignalAction.SHORT, symbol="AAA", price=price, stop=stop,
                  target=target, reasons=["test"], quality_score=3.0, rr_ratio=2.0)


# ---------------------------------------------------------------------------
# 1) Asıl kusur: fiyat stop'un ötesine geçmişken İŞLEM AÇILMAZ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("sig", "canli"), [(_long(), 97.5), (_short(), 102.5)])
def test_fiyat_stopu_gecmisse_girilmez(sig, canli):
    """GAP_STOP sınıfının kökü: canlı fiyat zaten stop'un ötesindeyken pozisyon açmak,
    açılır açılmaz −1R'den KÖTÜ kapanmak demektir (ölçülen ort. −1,497R)."""
    yeni, why = reanchor_to_live(sig, canli, min_rr=1.5)
    assert yeni is None
    assert "STALE_PRICE_THROUGH_STOP" in why


@pytest.mark.parametrize(("sig", "canli"), [(_long(), 98.0), (_short(), 102.0)])
def test_fiyat_tam_stopta_da_girilmez(sig, canli):
    """Sınır: risk tam 0 → RR tanımsız. Bölme hatası değil, RED verilmeli."""
    yeni, why = reanchor_to_live(sig, canli, min_rr=1.5)
    assert yeni is None and "STALE_PRICE_THROUGH_STOP" in why


def test_kacmis_kurulum_rr_kapisindan_dusar():
    """Fiyat LEHTE kaçtıysa da kurulum bayattır: hedef yakınlaştı, stop uzaklaştı →
    RR çöker. Kırılmayı kovalamak tam olarak budur."""
    yeni, why = reanchor_to_live(_long(), 103.0, min_rr=1.5)   # RR = 1/5 = 0.2
    assert yeni is None and "STALE_RR" in why


def test_gecerli_kurulum_canli_fiyata_capalanir():
    yeni, why = reanchor_to_live(_long(), 100.5, min_rr=1.5)   # RR = 3.5/2.5 = 1.4? hayır 1.4<1.5
    assert yeni is None and "STALE_RR" in why                  # sınır davranışı da kilitli
    yeni, why = reanchor_to_live(_long(), 100.0, min_rr=1.5)   # RR = 4/2 = 2.0
    assert yeni is not None and why == ""
    assert yeni.price == 100.0 and yeni.rr_ratio == pytest.approx(2.0)
    # Stop/target YAPISALDIR — kaydırılmaz (kaydırılsaydı RR sahte biçimde korunurdu).
    assert (yeni.stop, yeni.target) == (98.0, 104.0)
    assert any("LIVE_ANCHOR" in r for r in yeni.reasons)


def test_rr_gercek_fiyattan_yeniden_hesaplanir():
    """Sinyal RR'ı 2,0 diyordu; canlı fiyatta gerçek RR farklıdır ve YAZILAN o olmalı."""
    yeni, _ = reanchor_to_live(_long(price=100.0), 99.0, min_rr=1.0)
    assert yeni.rr_ratio == pytest.approx(5.0)   # (104-99)/(99-98)
    assert yeni.price == 99.0


def test_short_tarafi_dogru_isaretlenir():
    yeni, _ = reanchor_to_live(_short(), 101.0, min_rr=1.0)
    assert yeni is not None and yeni.rr_ratio == pytest.approx(5.0)   # (101-96)/(102-101)


# ---------------------------------------------------------------------------
# 2) Ölçüm YOKLUĞU bir RED gerekçesi değildir
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bozuk", [0.0, -1.0, float("nan"), None, "x"])
def test_kotasyon_yoksa_sinyal_degismeden_gecer(bozuk):
    """Kotasyon okunamadıysa eski davranış korunur — yoksa veri arızası tüm sinyalleri
    susturur (Tur-31: ölçüm yokluğu ölçüm gibi kullanılamaz)."""
    sig = _long()
    yeni, why = reanchor_to_live(sig, bozuk, min_rr=1.5)
    assert yeni is sig and why == ""


@pytest.mark.parametrize("action", [SignalAction.WAIT, SignalAction.EXIT])
def test_giris_disi_sinyal_dokunulmaz(action):
    sig = Signal(action=action, symbol="AAA", price=100.0, stop=98.0, target=104.0)
    yeni, why = reanchor_to_live(sig, 90.0, min_rr=1.5)
    assert yeni is sig and why == ""


# ---------------------------------------------------------------------------
# 3) Üretim çağıranı var mı? (yazılıp çağrılmayan onarım = onarım yok)
# ---------------------------------------------------------------------------
def test_scan_symbol_capayi_kullaniyor():
    src = _MAIN.read_text(encoding="utf-8")
    assert "reanchor_to_live(" in src, "onarım üretimde ÇAĞRILMIYOR"
    assert 'log_block("STALE_SIGNAL"' in src, "bayat kurulum girişi durdurmuyor"
    # Çapa UYARIDAN ÖNCE olmalı: operatöre bildirilen fiyat işlem görülebilir fiyat olmalı.
    assert src.index("reanchor_to_live(") < src.index("SIGNAL_ALERT_ALWAYS and valid_entry")


def test_capa_env_ile_kapatilabilir():
    """A/B ölçümü için eski (bayat-fiyat) davranışa dönüş yolu açık kalmalı."""
    from nihai.config import LIVE_ENTRY_ANCHOR
    assert LIVE_ENTRY_ANCHOR is True          # varsayılan: onarım AÇIK
    assert "LIVE_ENTRY_ANCHOR" in _MAIN.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 4) İkinci kaçak: zaman-stopu BAR değil GÜNCELLEME sayıyordu (birim onarımı)
#    Ölçülen: 510 işlemin 405'i TIME_STOP ile ~17 dk'da kapandı; kazananları 2R
#    hedefin çok altında +0,14R'de kesildi. Hedef ile ufuk aynı işlemi tarif etmiyordu.
# ---------------------------------------------------------------------------
from datetime import datetime, timedelta  # noqa: E402

from nihai.config import CONFIG  # noqa: E402
from nihai.risk.position_manager import PortfolioState, PositionManager, SlippageModel  # noqa: E402

_GIRIS = datetime(2024, 1, 2, 10, 30)     # RTH içi, seans sonundan uzak


def _pmgr_ile_pozisyon(*, entry=100.0, stop=95.0, target=110.0):
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    pmgr = PositionManager(ps, SlippageModel(stock_slippage_pct=0.0, option_slippage_pct=0.0))
    sig = Signal(action=SignalAction.LONG, symbol="AAA", price=entry, stop=stop,
                 target=target, rr_ratio=2.0)
    pmgr.open_from_signal(sig, {"shares": 100, "risk_dollars": 500}, now=_GIRIS,
                          asset_type="stock")
    return pmgr, next(iter(pmgr._trackers.values()))


def _cikis(pmgr, tr, *, px: float, sonra_dk: int):
    """Fiyat giriş bandında (yatay) → yalnız zaman/seans kuralları konuşur."""
    return pmgr.check_exit(tr, bar_high=px, bar_low=px, bar_open=px, bar_close=px,
                           now_dt=_GIRIS + timedelta(minutes=sonra_dk))


def test_canlida_12_guncelleme_12_dakikada_kapatmaz():
    """ASIL KUSUR: canlı döngüde 12 quote yoklaması ~12 dakikadır, 12 bar DEĞİL.
    Eski kural burada kapatıyordu; 1h geometriden türeyen 2R hedefi daha yolun başındaydı."""
    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = CONFIG.risk.time_stop_bars      # 12 "bar" = 12 güncelleme
    assert _cikis(pmgr, tr, px=100.1, sonra_dk=12) is None


def test_ufuk_dolunca_zaman_stopu_calisir():
    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = CONFIG.risk.time_stop_bars
    res = _cikis(pmgr, tr, px=100.1, sonra_dk=CONFIG.risk.time_stop_minutes + 5)
    assert res is not None and res[0] == "TIME_STOP"


def test_asla_eskisinden_ERKEN_tetiklenmez():
    """Değişimin yönü TEK TARAFLI: kural yalnız geciktirir. `bars_held` eşiği dolmadan
    saatler geçse bile zaman-stopu çalışmaz (ölçülmemiş bir hızlanma getirilmedi)."""
    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = CONFIG.risk.time_stop_bars - 1
    assert _cikis(pmgr, tr, px=100.1, sonra_dk=600) is None


def test_backtest_paritesi_korunur():
    """Backtest'te her güncelleme GERÇEK bir bardır: günlük barlarda 12. bar = 17.280 dk,
    dakika koşulu kendiliğinden sağlanır → eşik ESKİSİYLE AYNI barda tetiklenir."""
    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = CONFIG.risk.time_stop_bars
    res = _cikis(pmgr, tr, px=100.1, sonra_dk=12 * 1440)
    assert res is not None and res[0] == "TIME_STOP"


def test_seans_sonunda_duzlesir_kar_zarar_farketmez():
    """Gecelik taşıma YOK: kapanışa dakikalar kala pozisyon kâr/zarar durumundan
    BAĞIMSIZ kapatılır (gap riski pozisyonun yönünü sormaz)."""
    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = 1
    kapanisa_yakin = datetime(2024, 1, 2, 15, 55)     # 16:00 kapanışa 5 dk
    res = pmgr.check_exit(tr, bar_high=100.1, bar_low=100.1, bar_open=100.1,
                          bar_close=100.1, now_dt=kapanisa_yakin)
    assert res is not None and res[0] == "SESSION_FLAT"


def test_seans_ortasinda_duzlesme_yok():
    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = 1
    assert _cikis(pmgr, tr, px=100.1, sonra_dk=30) is None


def test_duzlesme_kapatilabilir(monkeypatch):
    """`FLAT_BEFORE_CLOSE_MIN=0` gecelik taşımaya bilinçli dönüş yoludur.
    CONFIG donmuş (frozen) → kopyasını yerine koyarız (üretim nesnesi değişmez)."""
    from dataclasses import replace

    import nihai.risk.position_manager as pm

    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = 1
    monkeypatch.setattr(
        pm, "CONFIG", replace(CONFIG, risk=replace(CONFIG.risk, flat_before_close_min=0)))
    res = pmgr.check_exit(tr, bar_high=100.1, bar_low=100.1, bar_open=100.1,
                          bar_close=100.1, now_dt=datetime(2024, 1, 2, 15, 55))
    assert res is None


def test_piyasa_kapaliyken_duzlesme_kurali_uygulanmaz():
    """`_minutes_to_close` kapalı seansta None döner → kural sessizce devre dışı kalır
    (ölçüm yokluğu bir çıkış gerekçesi değildir)."""
    pmgr, _tr = _pmgr_ile_pozisyon()
    assert pmgr._minutes_to_close(datetime(2024, 1, 6, 12, 0)) is None      # cumartesi
    assert pmgr._minutes_to_close("bozuk") is None


def test_stop_ve_target_duzlesmeden_ONCE_gelir():
    """Öncelik sırası: gerçek bir stop/target seans-sonu kuralına kurban edilmez."""
    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = 1
    res = pmgr.check_exit(tr, bar_high=111.0, bar_low=100.0, bar_open=100.0,
                          bar_close=111.0, now_dt=datetime(2024, 1, 2, 15, 55))
    assert res is not None and res[0] == "TARGET"


def test_zaman_okunamazsa_eski_yola_duser():
    """Bozuk/okunamaz giriş zamanı kuralı ÇÖKERTMEZ: yedek yol `bars_held`tir."""
    pmgr, tr = _pmgr_ile_pozisyon()
    tr.bars_held = CONFIG.risk.time_stop_bars
    tr.pos.entry_time = "bozuk"          # datetime değil
    res = pmgr.check_exit(tr, bar_high=100.1, bar_low=100.1, bar_open=100.1,
                          bar_close=100.1, now_dt=_GIRIS + timedelta(minutes=20))
    assert res is not None and res[0] == "TIME_STOP"
