"""Tur-58 — hub köprüsü: MainHub sürüklenme KAPISI + `/asset_flags` yolu.

İki ayrı kusuru kilitler:

1. **Sürüklenme (dış rapor §2.B).** `MainHub/` üç sağlayıcı dosyasını `nihai/data/` ile
   kopyala-yapıştır paylaşıyordu ve Tur-57 düzeltmeleri (kolon korunması, OPRA, bayraklar)
   MainHub kopyasına HİÇ ulaşmamıştı. Artık kopya ÜRETİLİR; bu test üretimin güncel
   olduğunu doğrular. MainHub'da elle yapılan düzenleme burada kırmızı verir.

2. **`asset_flags` hub üzerinden okunamıyordu.** `DATA_PROVIDER=hub` iken zincirde bu
   metodu taşıyan hiçbir düğüm yoktu → modül yedeği her tüketici süreçte AYRI bir Alpaca
   trading istemcisi kuruyordu (hub'ın tek-upstream amacının delinmesi). Uç + istemci
   metodu eklendi; buradaki testler yolun uçtan uca çalıştığını VE ölçüm-yokluğu
   sözleşmesinin (None ≠ False) korunduğunu doğrular.
"""
from __future__ import annotations

import pytest

from nihai.data import data_provider as dp
from tools import sync_mainhub


# --------------------------------------------------------------------------
# 1) sürüklenme kapısı
# --------------------------------------------------------------------------
@pytest.mark.skipif(not sync_mainhub.MAINHUB.is_dir(), reason="MainHub ağacı yok")
def test_mainhub_kopyalari_guncel():
    """MainHub kopyaları `nihai/data/`'dan üretilmiş hâliyle AYNI olmalı.

    Kırmızıysa düzeltme MainHub'da değil `nihai/` tarafında yapılır, sonra:
        python -m tools.sync_mainhub --apply
    """
    drifted = sync_mainhub.generate(apply=False)
    assert drifted == [], (
        f"MainHub sürüklendi: {drifted}. `python -m tools.sync_mainhub --apply` koş "
        f"(kaynak nihai/data/, MainHub kopyası ÜRETİLMİŞ dosyadır)."
    )


def test_rewrite_import_yollarini_duzeltir():
    """SAF yeniden yazım: paket yolları düz modül adlarına iner, gövde DEĞİŞMEZ."""
    src = (
        "from nihai.config import INTERVAL_MINUTES\n"
        "from nihai.data.data_provider import DataProvider\n"
        "    from nihai.ai.data_health import health_check\n"
        "x = 1  # nihai.data.foo bir yorumdur, dokunulmaz\n"
    )
    out = sync_mainhub.rewrite(src, "x")
    assert "from config import INTERVAL_MINUTES" in out
    assert "from data_provider import DataProvider" in out
    assert "    from data_health import health_check" in out       # girinti korunur
    assert "# nihai.data.foo bir yorumdur" in out                  # yorum bozulmaz
    assert "nihai." not in out.split("x = 1")[0].replace("nihai/data", "")


def test_rewrite_uretilmis_dosya_uyarisi_tasir():
    out = sync_mainhub.rewrite("x = 1\n", "data_provider")
    assert "ELLE DÜZENLEME" in out and "sync_mainhub --apply" in out


@pytest.mark.skipif(not sync_mainhub.MAINHUB.is_dir(), reason="MainHub ağacı yok")
def test_mainhub_tur57_duzeltmelerini_tasiyor():
    """Sürüklenmenin SOMUT sonucu: MainHub Tur-57'nin üç düzeltmesini de taşımalı."""
    ap = (sync_mainhub.MAINHUB / "alpaca_provider.py").read_text(encoding="utf-8")
    assert '("trade_count", "vwap")' in ap, "bar kolonları hâlâ düşürülüyor"
    assert "def asset_flags" in ap, "borç bayrakları MainHub'a ulaşmamış"
    assert "ALPACA_BARS_FEED" in ap and "enrich_chain_side" in ap


@pytest.mark.skipif(not sync_mainhub.MAINHUB.is_dir(), reason="MainHub ağacı yok")
def test_mainhub_hub_server_asset_flags_ucunu_sunuyor():
    """Uç İKİ hub_server'da da olmalı — biri eksikse `hub` kipi kaynağa göre ayrışır."""
    for path in (sync_mainhub.MAINHUB / "hub_server.py",
                 sync_mainhub.ROOT / "nihai" / "data" / "hub_server.py"):
        s = path.read_text(encoding="utf-8")
        assert '@app.get("/asset_flags")' in s, f"{path.name} /asset_flags sunmuyor"


# --------------------------------------------------------------------------
# 2) hub üzerinden asset_flags
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _hub(monkeypatch, payload, calls=None):
    h = dp.HubProvider.__new__(dp.HubProvider)      # __init__ httpx istemcisi kurar
    h.base_url = "http://test"
    h.timeout = 1.0

    def fake_get(path, params):
        if calls is not None:
            calls.append((path, params))
        return payload
    h._get = fake_get                                # type: ignore[method-assign]
    return h


def test_hub_asset_flags_ucu_cagriliyor(monkeypatch):
    calls: list = []
    h = _hub(monkeypatch, _FakeResp({"symbol": "AAPL", "shortable": True}), calls)
    assert h.asset_flags("aapl") == {"symbol": "AAPL", "shortable": True}
    assert calls == [("/asset_flags", {"symbol": "AAPL"})]


def test_hub_asset_flags_bos_govde_none(monkeypatch):
    """Hub `{}` döndürürse cevap None'dır — boş sözlük "satılabilir DEĞİL" demek değildir."""
    assert _hub(monkeypatch, _FakeResp({}), None).asset_flags("AAPL") is None


def test_hub_asset_flags_kesici_acikken_none(monkeypatch):
    assert _hub(monkeypatch, None, None).asset_flags("AAPL") is None


def test_hub_asset_flags_bozuk_json_none(monkeypatch):
    class _Bad:
        def json(self):
            raise ValueError("bozuk")
    assert _hub(monkeypatch, _Bad(), None).asset_flags("AAPL") is None


# --------------------------------------------------------------------------
# 3) zincir gezintisi: ölçüm YOKLUĞU aramayı bitirmemeli
# --------------------------------------------------------------------------
class _Nothing:
    """Metodu var ama ölçemiyor (hub kapalı) — zincir DEVAM etmeli."""
    def asset_flags(self, symbol):
        return None


class _Knows:
    def asset_flags(self, symbol):
        return {"symbol": symbol, "shortable": False, "easy_to_borrow": False}


class _Wrap:
    def __init__(self, inner):
        self.inner = inner


class _WrapChain:
    """dış düğüm ölçemiyor, iç düğüm biliyor."""
    def __init__(self):
        self.inner = _Knows()

    def asset_flags(self, symbol):
        return None


def test_zincir_bos_cevapta_devam_eder(monkeypatch):
    """Tur-58 regresyon kilidi: bir uç EKLEMEK mevcut bir yolu kapatmamalı.

    Eski kural "metodu taşıyan İLK düğümün cevabını döndür" idi; HubProvider bu metodu
    kazanınca hub kapalıyken (None) modül-düzeyi Alpaca yedeği ERİŞİLEMEZ oluyordu."""
    monkeypatch.setattr(dp, "_FLAG_PROVIDER", None, raising=False)
    got = dp.asset_flags("AAPL", _Wrap(_Knows()))
    assert got is not None and got["symbol"] == "AAPL"

    # ölçemeyen düğüm zinciri BİTİRMEZ → içteki bilen düğüme ulaşılır
    got2 = dp.asset_flags("MSFT", _WrapChain())
    assert got2 is not None and got2["symbol"] == "MSFT"


def test_hicbiri_bilmiyorsa_none(monkeypatch):
    """Anahtarsız ortamda dürüst cevap None'dır — False (= "satılamaz") DEĞİL."""
    # Yedek sağlayıcı ENJEKTE edilir: gerçek AlpacaProvider kurmak testi ağa çıkarırdı
    # (anahtar .env'den geliyor) ve hermetiklik kuralını bozardı.
    monkeypatch.setattr(dp, "_FLAG_PROVIDER", _Nothing(), raising=False)
    assert dp.asset_flags("AAPL", _Wrap(_Nothing())) is None
