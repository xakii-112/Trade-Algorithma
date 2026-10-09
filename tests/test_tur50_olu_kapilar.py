"""Tur-50 — ÖLÜ KAPILAR: var olan, belgelenen, ama HİÇBİR KOD YOLUNDAN OKUNMAYAN kurallar.

Kusur tek bir satırda değil bir SINIFTAYDI: `config.py` kendini ayar yüzeyi diye sunar
(126 alan, her biri PDF madde numarasıyla), `risk/` ise kural katmanı diye. İkisinde de
okuyanı OLMAYAN öğeler birikmişti:

* `OptionFilters.min_dte=5 / max_dte=60` — hiçbir yerden okunmuyordu ve YÜRÜRLÜKTEKİ
  pencereyle (`OptionStrategyConfig.dte_min=30 / dte_max=45`) ÇELİŞİYORDU. "0DTE'ye
  dokunma" yazan alanı sıkılaştıran operatör hiçbir şey değiştirmiş olmazdı.
* `OptionStrategyConfig.close_before_expiry_dte_override` — yorumu bir yedekleme zinciri
  TARİF EDİYORDU (`None → CONFIG.options.close_before_expiry_dte`), ama `check_exit`
  global alanı doğrudan okuyordu: override verilse SESSİZCE yok sayılırdı.
* `StrategyFilters.htf_bar_minutes=60` — türetilmiş bir değerin (INTERVAL_MINUTES
  [HTF_INTERVAL]) ikinci kopyasıydı; bir yorum operatörden ikisini elle "uyumlu
  tutmasını" istiyordu.
* `Regulation.finra_maintenance_margin=0.25` — uygulanmayan bir regülasyon sabiti.
* `RiskManager.margin_ok` — Madde 13'ün ADI KONMUŞ marj kapısı, ÇAĞIRANI YOK.
* `CircuitBreaker.trip/reset` — Madde 47'nin acil müdahale kancası, ÇAĞIRANI YOK.

Sonuncusu yalnız ölü değil ZARARLIYDI. `can_emit` hız sınırında `_tripped=True` kuruyordu;
hız sınırı GEÇİCİDİR (pencere kendi temizlenir) ama mandal KALICIDIR ve `reset()` hiçbir
yerden çağrılmadığı için süreç ömrü boyunca kapalı kalıyordu. Canlı karar loglarında
ölçüldü: **6 gerçek tetikleme ↔ 9.194 mandallı blok (%99,93)**; 2026-07-07'de tek mandal
17:39:56 → 19:47:25 arası 2.815 sinyali sustururken koşul 60 saniyede geçmişti.

Bu depo aynı sınıfı ELLE iki kez kapatmıştı — `risk_manager.portfolio_caps_breach`'in
kendi yorumları söylüyor: *"daha önce config'de tanımlı ama uygulanmıyordu"* ve
*"config'de ölüydü"*. Kural MAKİNEYE çevrilmediği için geri büyüdü. Bu dosya kuralı
makineye çevirir (Tur-24 deseni).

TARAMA KODA BAKAR, ANLATIYA DEĞİL (Tur-49 dersi): bu turun onarım YORUMLARI silinen
alanların adlarını bilerek ALINTILIYOR. Ham metinde arayan bir kapı kendi açıklamasına
takılır. Bu yüzden tarayıcı `ast` kullanır — yorumlar ve dizgi sabitleri ayrıştırma
sonucunda zaten yoktur.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nihai.config import CONFIG
from nihai.risk.safety import CircuitBreaker

_KOK = Path(__file__).resolve().parents[1]
_CONFIG_PY = _KOK / "nihai" / "config.py"


# ---------------------------------------------------------------------------
# Tarama altyapısı — AST (yorum/dizgi YOK), depo geneli
# ---------------------------------------------------------------------------
def _py_dosyalari(*altlar: str) -> list[Path]:
    """Verilen ağaçlardaki analiz edilecek kaynaklar (.venv ve önbellekler hariç).

    Tur-62: varsayılan artık YALNIZ ÜRETİM (`nihai`, `research`). Eskiden `tests` de
    dahildi ve bu, ölü-kapı taramasını KENDİ testinin tatmin etmesine yol açıyordu —
    bkz. `_okunan_adlar` ve `tests/test_tur62_test_kapisi_saymaz.py`."""
    out: list[Path] = []
    for alt in (altlar or ("nihai", "research")):
        for p in (_KOK / alt).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return out


def _agac(p: Path) -> ast.Module | None:
    try:
        return ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return None


def _okunan_adlar(*altlar: str) -> set[str]:
    """ÜRETİMDE erişilen her ad: `x.attr`, çıplak `ad` ve dizgi sabitleri (getattr/asdict).

    Dizgi sabitleri BİLEREK dahildir: `getattr(cfg, "meta_gate_mode")` gerçek bir
    okumadır. Yorumlar dahil DEĞİLDİR — `ast` onları zaten atar.

    **Tur-62 — `tests` bu kümeden ÇIKARILDI.** İlk yazımda dahildi ve kapı kendi kendini
    tatmin ediyordu: bir kuralın VAR olduğunu kanıtlamak için yazılan test, o kuralın
    ÇAĞRILDIĞININ de delili sayılıyordu. `RiskManager.pdt_blocks_day_trade` (Madde 10/23,
    README'de uygulama olarak gösterilir) tam olarak böyle hayatta kaldı: üretimde tek
    çağıranı yoktu, ama `test_risk_manager.py` onu çağırdığı için kapı yeşil kaldı."""
    adlar: set[str] = set()
    for p in _py_dosyalari(*altlar):
        t = _agac(p)
        if t is None:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.Attribute):
                adlar.add(node.attr)
            elif isinstance(node, ast.Name):
                adlar.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                adlar.add(node.value)
    return adlar


def _config_alanlari() -> dict[str, str]:
    """config.py'deki dataclass alanları → sahip sınıf adı."""
    t = _agac(_CONFIG_PY)
    assert t is not None, "config.py ayrıştırılamadı"
    alanlar: dict[str, str] = {}
    for node in ast.walk(t):
        if isinstance(node, ast.ClassDef):
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    alanlar[st.target.id] = node.name
    return alanlar


def _sinif_metotlari(p: Path, sinif: str) -> set[str]:
    t = _agac(p)
    assert t is not None
    for node in ast.walk(t):
        if isinstance(node, ast.ClassDef) and node.name == sinif:
            return {st.name for st in node.body
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()


def _fonksiyon_govdesi(p: Path, ad: str) -> ast.AST | None:
    t = _agac(p)
    assert t is not None
    for node in ast.walk(t):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == ad:
            return node
    return None


# ---------------------------------------------------------------------------
# KAPI 1 — hiçbir kod yolundan okunmayan config alanı KALMAMALI
# ---------------------------------------------------------------------------
# Bilinçli istisnalar. Boş bırakıldı: bugün ölü alan YOK. Bir alan buraya eklenirken
# NEDEN okunmadığı yazılmalıdır — liste, gerekçesiz büyümesin diye vardır.
_OLU_ALAN_IZNI: dict[str, str] = {}


def test_config_alanlarinin_hepsinin_bir_OKUYANI_var():
    """Asıl kusur buydu: 126 alanın 5'i hiçbir kod yolundan okunmuyordu."""
    alanlar = _config_alanlari()
    assert len(alanlar) > 100, f"config tarama bozuk: yalnız {len(alanlar)} alan bulundu"
    okunan = _okunan_adlar()
    olu = {ad: sinif for ad, sinif in alanlar.items()
           if ad not in okunan and ad not in _OLU_ALAN_IZNI}
    assert not olu, (
        "config.py'de OKUYANI OLMAYAN alan(lar) var — bir ayar, onu okuyan kod yoksa "
        f"ayar değil YANLIŞ BİLGİDİR: {sorted(f'{s}.{a}' for a, s in olu.items())}. "
        "Ya alanı okuyan yolu bağla, ya alanı sil, ya da _OLU_ALAN_IZNI'ne GEREKÇESİYLE ekle."
    )


def test_silinen_DTE_ciftinin_yerine_YURURLUKTEKI_pencere_var():
    """`OptionFilters.min_dte/max_dte` geri gelmemeli; pencere tek yerde yaşamalı."""
    alanlar = _config_alanlari()
    for olu_ad in ("min_dte", "max_dte", "htf_bar_minutes", "finra_maintenance_margin"):
        assert olu_ad not in alanlar, (
            f"`{olu_ad}` config'e ALAN olarak geri eklenmiş. Tur-50'de silinmişti çünkü "
            "hiçbir kod onu okumuyordu."
        )
    # Yürürlükteki pencere DURUYOR ve okunuyor.
    assert CONFIG.option_strategy.dte_min < CONFIG.option_strategy.dte_max


def test_DTE_penceresini_UYGULAYAN_yer_tektir():
    """Pencere `option_strategy.pick_expiry`'de uygulanır; ikinci bir uygulayan olmamalı."""
    kaynak = (_KOK / "nihai" / "risk" / "option_strategy.py")
    t = _agac(kaynak)
    assert t is not None
    okunan_burada = {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute)}
    assert {"dte_min", "dte_max"} <= okunan_burada


# ---------------------------------------------------------------------------
# KAPI 2 — para yolundaki (nihai/risk) public tanımların ÇAĞIRANI olmalı
# ---------------------------------------------------------------------------
# Bilinçli istisnalar — HER BİRİ İÇİN NEDEN yazılmalıdır (liste gerekçesiz büyümesin).
_URETIMSIZ_IZIN: dict[str, str] = {
    "assignment_risk_short_call":
        "Madde 28, YAZAN taraf. Bu sistem opsiyon YAZMAZ (tek bacak LONG call/put) → "
        "canlı çağıranı olamaz; Tur-58 bunu ÖLÇTÜ ve fonksiyonun kendi docstring'ine "
        "yazdı. Çok-bacaklı/kredi yapıları eklenirse kısa bacağın kapısı budur.",
}


def test_risk_katmaninda_cagirani_olmayan_public_tanim_yok():
    """`margin_ok` (Madde 13) ve `trip`/`reset` (Madde 47) tam olarak böyle ölmüştü.

    **Tur-62'de kapı İKİ yerden genişledi**, çünkü ikisi de gerçek kusur gizliyordu:

    1. Çağıran kümesi `tests`i de sayıyordu → bir kuralın var olduğunu KANITLAMAK için
       yazılan test, o kuralın ÇAĞRILDIĞININ delili oluyordu. `pdt_blocks_day_trade`
       (Madde 10/23) ve `beta_hedge_suggestion` (BULGU-59-6) böyle hayatta kaldı.
    2. Yalnız SINIF METOTLARI taranıyordu; modül düzeyi public fonksiyonlar hiç
       bakılmamıştı — `options_analyzer.assignment_risk_short_call` oradaydı."""
    okunan = _okunan_adlar()
    olu: list[str] = []

    def _bak(ad: str, konum: str) -> None:
        if ad.startswith("_") or ad in okunan or ad in _URETIMSIZ_IZIN:
            return
        olu.append(konum)

    for p in (_KOK / "nihai" / "risk").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        t = _agac(p)
        if t is None:
            continue
        # Modül düzeyi public fonksiyonlar (Tur-62'de eklendi)
        for st in t.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _bak(st.name, f"{p.name}:{st.lineno} {st.name}")
        # Sınıf metotları
        for node in ast.walk(t):
            if isinstance(node, ast.ClassDef):
                for st in node.body:
                    if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    dekor = {ast.unparse(d) for d in st.decorator_list}
                    if any(("property" in d or "setter" in d) for d in dekor):
                        continue
                    _bak(st.name, f"{p.name}:{st.lineno} {node.name}.{st.name}")
    assert not olu, (
        "nihai/risk içinde ÜRETİMDEN ÇAĞRILMAYAN public tanım(lar): "
        f"{sorted(olu)}. Bir kural, onu çağıran ÜRETİM yolu yoksa kural DEĞİLDİR — "
        "ve yalnız kendi testi çağırıyorsa kapı kendini tatmin ediyor demektir."
    )


# ---------------------------------------------------------------------------
# KAPI 3 — hız sınırı MANDALLAMAZ (asıl kusur)
# ---------------------------------------------------------------------------
def test_hiz_siniri_mandallamaz_pencere_daralinca_sinyal_yeniden_akar(monkeypatch):
    """6 gerçek tetikleme → 9.194 mandallı blok kusurunun sözleşme testi."""
    from datetime import datetime, timedelta

    import nihai.risk.safety as safety

    cb = CircuitBreaker()
    limit = CONFIG.safety.max_signals_per_minute

    simdi = datetime(2026, 8, 27, 18, 0, 0)

    class _SahteZaman(datetime):
        @classmethod
        def now(cls, tz=None):        # noqa: ARG003
            return simdi

    monkeypatch.setattr(safety, "datetime", _SahteZaman)

    for _ in range(limit):
        ok, _why = cb.can_emit()
        assert ok, "limitin ALTINDA reddetmemeli"
        cb.record_emit()

    ok, why = cb.can_emit()
    assert not ok and why.startswith("RATE_LIMIT_MIN_"), why
    assert not why.startswith("TRIPPED:"), "hız sınırı MANDAL kurmamalı"
    assert not cb.tripped, "hız sınırı devre kesiciyi TRİPLEMEMELİ (asıl kusur buydu)"

    # Pencere 61 sn sonra boşalır → sinyal KENDİLİĞİNDEN yeniden akar (mandal olsaydı akmazdı).
    simdi = simdi + timedelta(seconds=61)
    ok, why = cb.can_emit()
    assert ok, f"pencere daraldıktan sonra yayılabilmeliydi, why={why}"


def test_saatlik_sinir_da_mandallamaz(monkeypatch):
    """Saatlik sınır da GEÇİCİ bir koşuldur; kayıtlar 58 sn arayla → dakika sınırı temiz."""
    from datetime import datetime, timedelta

    import nihai.risk.safety as safety

    cb = CircuitBreaker()
    t0 = datetime(2026, 8, 27, 18, 0, 0)
    saat = {"t": t0}

    class _SahteZaman(datetime):
        @classmethod
        def now(cls, tz=None):        # noqa: ARG003
            return saat["t"]

    monkeypatch.setattr(safety, "datetime", _SahteZaman)

    n = CONFIG.safety.max_signals_per_hour
    for i in range(n):
        saat["t"] = t0 + timedelta(seconds=58 * i)      # 58 sn → per_min hep 1
        cb.record_emit()
    saat["t"] = t0 + timedelta(seconds=58 * n)
    ok, why = cb.can_emit()
    assert not ok and why.startswith("RATE_LIMIT_HOUR_"), why
    assert not cb.tripped, "saatlik sınır da MANDAL kurmamalı"

    # En eski kayıtlar 1 saatlik pencereden düşünce yayın yeniden açılır.
    saat["t"] = t0 + timedelta(hours=1, seconds=59)
    assert cb.can_emit()[0], "pencere kayınca saatlik sınır da kendiliğinden geçmeli"


def test_can_emit_govdesinde_MANDAL_atamasi_YOK():
    """Regresyon kapısı — kusurun kendi ifadesi. AST'ye bakar (yorumlar ayrıştırmada yok)."""
    gov = _fonksiyon_govdesi(_KOK / "nihai" / "risk" / "safety.py", "can_emit")
    assert gov is not None
    atananlar = {
        t.attr
        for node in ast.walk(gov)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute)
    }
    assert "_tripped" not in atananlar, (
        "`can_emit` yeniden `_tripped` kuruyor — hız sınırı GEÇİCİ bir koşuldur, "
        "mandal ise KALICI. Bu tam olarak Tur-50'de kapatılan kusurdur."
    )
    assert "_trip_reason" not in atananlar


# ---------------------------------------------------------------------------
# KAPI 4 — Madde 47 kancası ERİŞİLEBİLİR (trip/reset artık ölü değil)
# ---------------------------------------------------------------------------
def test_devre_kesici_mandali_ELLE_kurulur_ve_YALNIZ_reset_temizler():
    cb = CircuitBreaker()
    assert cb.can_emit()[0]
    cb.trip("MANUEL_TEST")
    ok, why = cb.can_emit()
    assert not ok and why == "TRIPPED: MANUEL_TEST"
    assert cb.tripped and cb.trip_reason == "MANUEL_TEST"
    cb.reset()
    assert not cb.tripped and cb.trip_reason == ""
    assert cb.can_emit()[0], "reset sonrası sinyal yeniden akmalı"


def test_operator_komutlari_tg_handlers_tablosunda_KAYITLI():
    """`trip`/`reset` bir yüzeye bağlanmadan Madde 47 kancası ERİŞİLEMEZDİR."""
    t = _agac(_KOK / "nihai" / "execution" / "main.py")
    assert t is not None
    anahtarlar: set[str] = set()
    for node in ast.walk(t):
        if isinstance(node, ast.Assign):
            for hedef in node.targets:
                if isinstance(hedef, ast.Name) and hedef.id == "tg_handlers":
                    if isinstance(node.value, ast.Dict):
                        anahtarlar = {
                            k.value for k in node.value.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)
                        }
    assert {"dur", "devam"} <= anahtarlar, (
        f"operatör dur/devam komutları tg_handlers'ta yok: {sorted(anahtarlar)}"
    )


def test_dur_komutu_TRIP_devam_komutu_RESET_cagirir():
    """Handler adı kayıtlı olsa da GÖVDESİ doğru çağrıyı yapmazsa kanca yine ölüdür."""
    yol = _KOK / "nihai" / "execution" / "main.py"
    for fn_ad, beklenen in (("_cmd_dur", "trip"), ("_cmd_devam", "reset")):
        gov = _fonksiyon_govdesi(yol, fn_ad)
        assert gov is not None, f"{fn_ad} tanımlı değil"
        cagrilar = {
            node.func.attr for node in ast.walk(gov)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert beklenen in cagrilar, f"{fn_ad} `breaker.{beklenen}()` çağırmıyor: {sorted(cagrilar)}"


def test_dur_komutu_ACIK_POZISYONLARA_dokunmaz():
    """Bir operatör freni zarar-kesmeyi geciktiremez: `/dur` yalnız YAYINI durdurur."""
    gov = _fonksiyon_govdesi(_KOK / "nihai" / "execution" / "main.py", "_cmd_dur")
    assert gov is not None
    cagrilar = {
        node.func.attr for node in ast.walk(gov)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    yasak = {"close_position", "close_all", "flatten", "check_exit", "on_close"}
    assert not (cagrilar & yasak), f"/dur çıkış yoluna dokunuyor: {sorted(cagrilar & yasak)}"


# ---------------------------------------------------------------------------
# KAPI 4b — hiçbir kaynak GEÇERSİZ ESCAPE taşımamalı
# ---------------------------------------------------------------------------
def test_hicbir_kaynak_gecersiz_escape_dizisi_tasimaz():
    """Bu turun kendi tarayıcısının ORTAYA ÇIKARDIĞI kusur.

    `nihai/console.py` docstring'i `logs\\*.log` yazıyordu; ham dizgi olmadığı için
    Python 3.12 `SyntaxWarning: invalid escape sequence '\\*'` üretiyor ve gelecek bir
    sürümde bu bir `SyntaxError` olacak. İroni kayda değer: uyarıyı üreten modül tam da
    CLI çıktısının GÜVENLİĞİNİ garanti etmek için yazılmış olandır.

    Ne `ruff` (W605 bu projenin `select` listesinde YOK: E,F,I,B,UP) ne `mypy` bakıyordu;
    318 dosyada tek örnekti."""
    import warnings

    kusurlu: list[str] = []
    # Bu kapı SÖZDİZİMİ tarar, çağrı değil → `tests` de dahil olmalı (Tur-62'de
    # varsayılan üretimle sınırlandığı için kökler burada AÇIKÇA verilir).
    for p in _py_dosyalari("nihai", "research", "tests"):
        with warnings.catch_warnings(record=True) as yakalanan:
            warnings.simplefilter("always")
            try:
                ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError as e:                     # pragma: no cover
                kusurlu.append(f"{p.name}:{e.lineno} {e.msg}")
                continue
            for uy in yakalanan:
                if issubclass(uy.category, SyntaxWarning):
                    kusurlu.append(f"{p.name}: {uy.message}")
    assert not kusurlu, (
        "geçersiz escape dizisi taşıyan kaynak(lar) — bugün SyntaxWarning, yarın "
        f"SyntaxError: {kusurlu}. Ham dizgi (r\"\"\") kullan ya da ters bölüyü kaçır."
    )


# ---------------------------------------------------------------------------
# KAPI 5 — DTE override zinciri artık GERÇEK
# ---------------------------------------------------------------------------
def test_dte_override_None_iken_davranis_BIT_DENK():
    """Varsayılan None → eşik global alandır (onarım davranışı değiştirmemeli)."""
    assert CONFIG.option_strategy.close_before_expiry_dte_override is None
    gov = _fonksiyon_govdesi(_KOK / "nihai" / "risk" / "position_manager.py", "check_exit")
    assert gov is not None
    okunan = {n.attr for n in ast.walk(gov) if isinstance(n, ast.Attribute)}
    assert "close_before_expiry_dte_override" in okunan, (
        "check_exit override'ı OKUMUYOR — yorumun tarif ettiği zincir yine sahte."
    )
    assert "close_before_expiry_dte" in okunan, "global yedek de okunmalı"


@pytest.mark.parametrize(("override", "dte", "cikis_bekleniyor"), [
    (None, 8, False),    # global eşik 5 → 8 DTE'de çıkış YOK (varsayılan davranış)
    (None, 4, True),     # global eşik 5 → tetikler
    (10, 8, True),       # override 10 → 8 DTE'de ARTIK tetikler (eskiden YOK SAYILIRDI)
    (2, 4, False),       # override 2 → global 5'ten gevşek; 4 DTE'de tetiklememeli
])
def test_override_esigi_URETIM_YOLUNDA_gercekten_DEGISTIRIR(
        monkeypatch, override, dte, cikis_bekleniyor):
    """Kusurun ölçülebilir hâli — `check_exit`'in KENDİSİ üzerinden.

    Aritmetiği testte yeniden yazmak kusurun geri gelmesini YAKALAMAZDI: üretim yolu
    global alanı okumaya dönse test yine yeşil kalırdı."""
    import dataclasses
    from datetime import datetime, timedelta

    import nihai.risk.position_manager as pm
    from nihai.risk.options_analyzer import GreekSet
    from nihai.risk.position_manager import PositionManager, SlippageModel
    from nihai.risk.risk_manager import PortfolioState
    from nihai.signals.strategy import Signal, SignalAction

    yeni_os = dataclasses.replace(CONFIG.option_strategy,
                                  close_before_expiry_dte_override=override)
    monkeypatch.setattr(pm, "CONFIG", dataclasses.replace(CONFIG, option_strategy=yeni_os))

    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    pmgr = PositionManager(ps, SlippageModel(stock_slippage_pct=0.0, option_slippage_pct=0.0))
    simdi = datetime.now()
    vade = (simdi + timedelta(days=dte)).date().isoformat()
    sig = Signal(action=SignalAction.LONG, symbol="AAA", price=2.0, stop=1.0, target=4.0)
    pmgr.open_from_signal(sig, {"contracts": 1, "risk_dollars": 200}, now=simdi,
                          asset_type="option", option_type="c", strike=100,
                          expiry=vade, greeks=GreekSet(0.5, 0.02, -0.05, 0.1, 0.3))

    tr = next(iter(pmgr._trackers.values()))
    res = pmgr.check_exit(tr, bar_high=2.0, bar_low=2.0, bar_open=2.0, bar_close=2.0,
                          now_dt=simdi)
    tetikledi = bool(res and res[0] == "DTE_GUARD")
    assert tetikledi is cikis_bekleniyor, (
        f"override={override} dte={dte} → DTE_GUARD={tetikledi}, beklenen={cikis_bekleniyor}"
    )
