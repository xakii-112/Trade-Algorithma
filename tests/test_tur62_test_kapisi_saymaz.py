"""Tur-62 — KENDİ TESTİ ÇAĞIRIYORSA KAPI HÂLÂ ÖLÜDÜR.

Tur-50 bu depoya "bir kural, onu çağıran yol yoksa kural DEĞİLDİR" kapısını koydu ve
`margin_ok` · `trip` · `reset` · `heartbeat`i o kapıyla yakaladı. Kapı doğruydu; **çağıran
kümesi** yanlıştı. `_okunan_adlar()` `nihai` + `research` + **`tests`** ağaçlarını
tarıyordu, yani:

    Bir kuralın VAR olduğunu KANITLAMAK için yazılan test,
    o kuralın ÇAĞRILDIĞININ da delili sayılıyordu.

Kapı böylece kendi kendini tatmin eder: kuralı yaz, testini yaz, kapı yeşil — üretimde tek
satır onu çağırmasa bile. Ölü kapıyı arayan tarayıcı, ölü kapıyı GİZLEYEN şeye dönüşür.

İkinci körlük: tarama YALNIZ sınıf metotlarına bakıyordu; `nihai/risk` altındaki modül
düzeyi public fonksiyonlar hiç görülmemişti.

ÖLÇÜLDÜ (üretim-yalnız çağıran kümesiyle, `nihai/` geneli): body'si üretimden hiç
çağrılmayan public tanım sayısı **5**. Tur-50'nin dürüstlük bölümü kapıyı tüm depoya
açmanın "onlarca meşru genel-API tanımını" yakalayacağını yazmıştı; ölçüm bunu
çürüttü — beşi de gerçek, ikisi tam da PARA YOLUNDA:

    nihai/risk/risk_manager.py  RiskManager.pdt_blocks_day_trade     (Madde 10, 23)
    nihai/risk/risk_manager.py  RiskManager.beta_hedge_suggestion    (BULGU-59-6)
    nihai/risk/options_analyzer.py  assignment_risk_short_call       (Madde 28 — İZİNLİ)
    nihai/ai/neural.py  NeuralEnsemble.member_spread                 (kütüphane API'si)
    nihai/ai/online.py  FTRLProximal.predict_one / update_one        (kütüphane API'si)

`config.py`nin 125 alanı için aynı sıkı taramayı koşturmak **negatif** sonuç verdi
(üretimde okunmayan alan: 0) — bu da bir ölçümdür ve kapının o kanadının sağlam olduğunu
söyler.

Bu dosya üç şeyi kilitler: (A) tarayıcının çağıran kümesi bir daha `tests`e açılamaz,
(B) PDT freni üretim yolundan GERÇEKTEN çalışır, (C) beta hedge önerisi panoya ulaşır ve
"ölçülemedi" ile "ihlal yok" ayrı kalır.
"""
from __future__ import annotations

import ast
import dataclasses
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import nihai.risk.risk_manager as rm
import nihai.web.web_dashboard as wd
from nihai.config import CONFIG
from nihai.risk.risk_manager import PortfolioState, RiskManager, VirtualPosition

_KOK = Path(__file__).resolve().parents[1]
_TUR50 = _KOK / "tests" / "test_tur50_olu_kapilar.py"
_RISK_MGR = _KOK / "nihai" / "risk" / "risk_manager.py"
_PANO_PY = _KOK / "nihai" / "web" / "web_dashboard.py"
_PANO_HTML = _KOK / "nihai" / "web" / "web_dashboard.html"

client = TestClient(wd.app)


def _agac(p: Path) -> ast.Module:
    return ast.parse(p.read_text(encoding="utf-8", errors="ignore"))


def _fonksiyon(p: Path, ad: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(_agac(p)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == ad:
            return node
    raise AssertionError(f"{p.name} içinde `{ad}` tanımı YOK")


def _cagrilan_adlar(node: ast.AST) -> set[str]:
    """Bir gövdenin ÇAĞIRDIĞI adlar (`x.f()` → 'f', `f()` → 'f'). Yorum/dizgi DEĞİL."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


def fresh(equity: float = 200_000.0) -> PortfolioState:
    return PortfolioState(equity=equity, ath_equity=equity, cash_balance=equity)


def _pdt_acik(monkeypatch) -> None:
    """PDT açık + günlük kayıp/soğutma frenleri KAPALI (kardeş frenler karışmasın)."""
    yeni = dataclasses.replace(
        CONFIG,
        regulation=dataclasses.replace(CONFIG.regulation, pdt_enabled=True),
        risk=dataclasses.replace(CONFIG.risk, daily_max_loss_pct=0.0,
                                 consecutive_stops_cooldown=0),
    )
    monkeypatch.setattr(rm, "CONFIG", yeni)


# ---------------------------------------------------------------------------
# KAPI A — tarayıcının ÇAĞIRAN kümesi bir daha `tests`e açılamaz
# ---------------------------------------------------------------------------
def test_tur50_tarayicisinin_varsayilan_kokleri_tests_ICERMEZ():
    """Asıl kusur buydu: kapı kendi testini çağıran sayıyordu.

    `ast` ile bakılır, metinle değil — bu dosyanın kendi açıklaması `"tests"` kelimesini
    zaten defalarca geçiriyor ve ham metin araması kendi anlatısına takılırdı (Tur-49)."""
    fn = _fonksiyon(_TUR50, "_py_dosyalari")
    varsayilanlar: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Tuple):
            degerler = [e.value for e in node.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            varsayilanlar.extend(degerler)
    assert varsayilanlar, "_py_dosyalari varsayılan kök demeti bulunamadı"
    assert "tests" not in varsayilanlar, (
        "Tur-50 tarayıcısının VARSAYILAN kökleri yine `tests` içeriyor. Bir kuralın "
        "var olduğunu kanıtlayan test, o kuralın ÇAĞRILDIĞININ delili OLAMAZ."
    )
    assert "nihai" in varsayilanlar, "üretim ağacı taranmıyor — kapı tamamen körelmiş"


def test_okunan_adlar_kokleri_py_dosyalarindan_ALIR():
    """`_okunan_adlar` kendi ağaç listesini kurarsa KAPI A atlatılabilirdi."""
    fn = _fonksiyon(_TUR50, "_okunan_adlar")
    assert "_py_dosyalari" in _cagrilan_adlar(fn), (
        "_okunan_adlar artık _py_dosyalari'nı çağırmıyor → kök listesi ikiye ayrıldı, "
        "biri düzeltilirken diğeri sessizce eski davranışta kalabilir."
    )


def test_uretimsiz_izin_listesi_her_giris_icin_GEREKCE_tasir():
    """İzin listesi, gerekçesiz büyümesin diye vardır (Tur-50 `_OLU_ALAN_IZNI` deseni)."""
    from tests.test_tur50_olu_kapilar import _URETIMSIZ_IZIN
    for ad, neden in _URETIMSIZ_IZIN.items():
        assert len(neden.strip()) >= 40, (
            f"`{ad}` izinli ama gerekçesi yok/yetersiz. Neden üretimden çağrılamadığı "
            "YAZILMALIDIR; aksi halde liste kapıyı öldürür."
        )


# ---------------------------------------------------------------------------
# KAPI B — PDT freni ÜRETİM yolundan gerçekten çalışır
# ---------------------------------------------------------------------------
def test_pdt_freni_giris_kapilari_demetinde_KAYITLI():
    """`_entry_blocked` gövdesi üç freni de ÇAĞIRMALI (metin değil, `ast`)."""
    cagrilar = _cagrilan_adlar(_fonksiyon(_RISK_MGR, "_entry_blocked"))
    for gerekli in ("daily_loss_breach", "consecutive_stops_cooldown", "pdt_blocks_day_trade"):
        assert gerekli in cagrilar, f"`_entry_blocked` `{gerekli}` frenini çağırmıyor"


@pytest.mark.parametrize("sizer", ["size_stock", "size_long_option"])
def test_her_iki_boyutlandirici_TEK_fren_demetinden_gecer(sizer):
    """Tur-28: bir kural iki kapıdan yalnız birine konursa yalnız birinde geçerlidir.

    Frenler eskiden HER İKİ gövdede kopyalanmıştı ve PDT ikisine de eklenmemişti. Artık
    tek çağrı var; kopyalama geri gelirse (gövde freni DOĞRUDAN çağırırsa) bu test kırılır
    ve bir sonraki fren yine tek yola düşer."""
    cagrilar = _cagrilan_adlar(_fonksiyon(_RISK_MGR, sizer))
    assert "_entry_blocked" in cagrilar, f"`{sizer}` fren demetini çağırmıyor"
    for kopya in ("daily_loss_breach", "consecutive_stops_cooldown", "pdt_blocks_day_trade"):
        assert kopya not in cagrilar, (
            f"`{sizer}` `{kopya}` frenini DOĞRUDAN çağırıyor — demet ikiye ayrılmış."
        )


def test_esik_ALTINDA_butce_dolunca_giris_REDDEDILIR(monkeypatch):
    """Kusurun kendi ifadesi: bu senaryo Tur-62'ye kadar `approved=True` dönüyordu."""
    _pdt_acik(monkeypatch)
    ps = fresh(10_000)                      # NLV < 25k → PDT penceresi geçerli
    ps.day_trades_count = CONFIG.regulation.pdt_max_day_trades_under_threshold
    ps.day_trades_window_start = date.today()
    rc = RiskManager(ps).size_stock(entry=100.0, stop=97.0, symbol="X")
    assert not rc.approved and rc.reason == "PDT_DAY_TRADE_LIMIT"


def test_esik_ALTINDA_butce_dolunca_OPSIYON_girisi_de_REDDEDILIR(monkeypatch):
    from nihai.risk.options_analyzer import GreekSet
    _pdt_acik(monkeypatch)
    ps = fresh(10_000)
    ps.day_trades_count = CONFIG.regulation.pdt_max_day_trades_under_threshold
    ps.day_trades_window_start = date.today()
    greeks = GreekSet(delta=0.5, gamma=0.01, theta=-0.05, vega=0.1, iv=0.30)
    rc = RiskManager(ps).size_long_option(2.0, greeks, symbol="X")
    assert not rc.approved and rc.reason == "PDT_DAY_TRADE_LIMIT"


def test_esik_USTUNDE_fren_ATILDIR_canli_hesap_BIT_DENK(monkeypatch):
    """Canlı hesabın ölçülen NLV aralığı 294.296$–645.264$ (eşiğin 11,8 katı).

    Bağlantı bu yüzden BİT-DENKtir: sayaç dolu olsa bile fren düşmez. Bu testin
    kırılması, canlı davranışın sessizce değiştiği anlamına gelir."""
    _pdt_acik(monkeypatch)
    ps = fresh(294_423.0)
    ps.day_trades_count = CONFIG.regulation.pdt_max_day_trades_under_threshold + 50
    rc = RiskManager(ps).size_stock(entry=100.0, stop=97.0, symbol="X")
    assert rc.approved and rc.reason == "OK"


def test_pdt_KAPALIYKEN_esik_alti_bile_engellenmez(monkeypatch):
    """Operatörün kapatma düğmesi gerçekten kapatmalı (Tur-44: kapı sessizce degrade olmaz)."""
    yeni = dataclasses.replace(
        CONFIG,
        regulation=dataclasses.replace(CONFIG.regulation, pdt_enabled=False),
        risk=dataclasses.replace(CONFIG.risk, daily_max_loss_pct=0.0,
                                 consecutive_stops_cooldown=0),
    )
    monkeypatch.setattr(rm, "CONFIG", yeni)
    ps = fresh(10_000)
    ps.day_trades_count = CONFIG.regulation.pdt_max_day_trades_under_threshold + 5
    rc = RiskManager(ps).size_stock(entry=100.0, stop=97.0, symbol="X")
    assert rc.approved and rc.reason == "OK"


def test_pdt_gunluk_sinir_FINRA_degerini_tasir():
    """BULGU-62-4 — sabit, ADINI TAŞIDIĞI kuralla çelişiyordu.

    Varsayılan **10**du; `risk_manager.pdt_blocks_day_trade` yorumu ise FINRA'nın gerçek
    değerini yazıyor: *"NLV < $25k → 5 iş günü içinde max 3 day-trade"*. 10'un hiçbir
    yerde gerekçesi yoktu (Tur-61'in belge↔kod sınıfı). Kural değeri bir TERCİH değildir.

    `PDT_MAX_DAY_TRADES_UNDER_THRESHOLD` ile gevşetilebilir; bu test env'i değil
    VARSAYILANI denetler, çünkü kusur varsayılandaydı."""
    import os
    if os.getenv("PDT_MAX_DAY_TRADES_UNDER_THRESHOLD"):
        pytest.skip("env ile bilinçli olarak geçersiz kılınmış")
    assert CONFIG.regulation.pdt_max_day_trades_under_threshold == 3, (
        "PDT günlük-işlem sınırı FINRA'nın 3'ünden farklı. Farklıysa GEREKÇESİ "
        "config.py'de yazmalıdır — adını taşıdığı kuralla çelişen bir sabit, "
        "uygulanmayan bir kuraldan daha kötüdür: uygulanıyor GİBİ görünür."
    )


def test_pdt_penceresi_kaynakta_TAKVIM_gunu_oldugunu_soyluyor():
    """Dürüstlük kapısı: kaynak, pencerenin NE saydığını doğru söylemelidir.

    Tur-62'de pencere `timedelta(days=...)` ile TAKVİM günü sayıyordu ve kapı bunun yazılı
    olmasını istiyordu. Tur-90 (BULGU-90-4): hesap eşiğin altına indi (19,5 k$) ve ponytail
    notunun koşulu doldu → pencere artık YÜRÜYEN son N NYSE seansıdır (`pdt_sessions`). Kapının
    sözleşmesi aynı kaldı (kaynak davranışı söyler); söylenen şey değişti."""
    kaynak = _RISK_MGR.read_text(encoding="utf-8")
    assert "NYSE seansı" in kaynak and "def pdt_sessions" in kaynak, (
        "PDT penceresinin yürüyen iş günü (NYSE seansı) saydığı kaynakta yazmıyor.")
    assert "timedelta(days=CONFIG.regulation.pdt_lookback_days)" not in kaynak, (
        "takvim-günü blok sıfırlaması geri gelmiş (Tur-90 regresyonu)")


def test_config_yorumu_artik_BAGLI_OLMADIGINI_soylemiyor():
    """Regresyon kapısı: kusurun kendi ifadesi kaynakta ARANIR (Tur-48 deseni)."""
    kaynak = (_KOK / "nihai" / "config.py").read_text(encoding="utf-8")
    assert "giriş yolundan ÇAĞRILMIYOR" not in kaynak, (
        "config.py hâlâ PDT'nin giriş yolundan çağrılmadığını yazıyor — belge ile kod "
        "yeniden ayrıştı (Tur-61 BULGU-61-4 sınıfı)."
    )


# ---------------------------------------------------------------------------
# KAPI C — beta hedge önerisi panoya ULAŞIR ve "ölçülemedi" ayrı kalır
# ---------------------------------------------------------------------------
def test_status_ucu_beta_hedge_ALANLARINI_tasir():
    r = client.get("/api/status")
    assert r.status_code == 200
    bh = r.json().get("beta_hedge")
    assert bh is not None, "/api/status `beta_hedge` taşımıyor — yüzey yine yetim"
    for alan in ("suggestion", "net_beta_dollars", "limit_dollars",
                 "measured", "unmeasured", "coverage"):
        assert alan in bh, f"`beta_hedge.{alan}` uçtan düşmüş (pano bu alanı okuyor)"


def test_panonun_okudugu_her_beta_alani_UCTA_var():
    """Backend↔frontend sözleşmesi (Tur-61 deseni): pano var olmayan anahtar okursa
    hata YOK, yalnız daima yanlış bir kart olur."""
    html = _PANO_HTML.read_text(encoding="utf-8")
    bh = client.get("/api/status").json()["beta_hedge"]
    for alan in ("measured", "unmeasured", "coverage", "net_beta_dollars",
                 "limit_dollars", "suggestion"):
        if f"bh.{alan}" in html:
            assert alan in bh, f"pano `bh.{alan}` okuyor ama uç bu alanı vermiyor"
    assert "beta-hedge-val" in html, "beta hedge kartı panodan kaybolmuş"


def test_beta_hedge_uretim_okuyucusu_VAR():
    """`beta_hedge_suggestion`ın çağıranı yeniden yalnız TEST olamaz (`ast` ile)."""
    cagrilar: set[str] = set()
    for node in ast.walk(_agac(_PANO_PY)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            cagrilar.add(node.func.attr)
    assert "beta_hedge_suggestion" in cagrilar, (
        "pano ucu `beta_hedge_suggestion`ı ÇAĞIRMIYOR — bir yorum/metin bu kapıyı "
        "tatmin edemesin diye çağrı `ast` ile aranır."
    )


def test_olculmemis_beta_ONERI_YOKLUGUNDAN_ayrilir(monkeypatch):
    """Tur-60 dersi: "ölçülemedi" ile "ihlal yok" tek etikete yıkılamaz.

    İki durumda da `suggestion is None` döner; ayıran alan `measured`tır."""
    ps = fresh(100_000.0)
    ps.open_positions.append(VirtualPosition(
        symbol="X", asset_type="stock", side="LONG", quantity=100,
        entry_price=100.0, entry_time=date.today(), beta=None))   # betası ÖLÇÜLMEMİŞ
    monkeypatch.setattr(wd, "get_portfolio_status", lambda: (wd.StateStore(), ps))
    bh = client.get("/api/status").json()["beta_hedge"]
    assert bh["suggestion"] is None
    assert bh["measured"] == 0 and bh["unmeasured"] == 1, (
        "ölçülmemiş pozisyon 'ihlal yok' gibi görünüyor — kart yeşil yalan söyler"
    )


def test_limiti_asan_beta_ONERI_uretir(monkeypatch):
    """Pozitif yol: net beta limiti aşınca ters bacak ÖNERİLİR (emir GÖNDERİLMEZ)."""
    ps = fresh(100_000.0)
    limit = 100_000.0 * CONFIG.portfolio.max_net_directional_pct
    # limitin ~2 katı net beta-dolar: nominal = qty × entry, beta = 2.0
    qty = int((limit * 2.0) / (100.0 * 2.0)) + 10
    ps.open_positions.append(VirtualPosition(
        symbol="X", asset_type="stock", side="LONG", quantity=qty,
        entry_price=100.0, entry_time=date.today(), beta=2.0))
    monkeypatch.setattr(wd, "get_portfolio_status", lambda: (wd.StateStore(), ps))
    bh = client.get("/api/status").json()["beta_hedge"]
    s = bh["suggestion"]
    assert s is not None, "limit aşıldı ama öneri yok"
    assert s["side"] == "SHORT"                     # uzun maruziyetin TERS bacağı
    assert s["notional"] > 0 and s["coverage"] == 1.0
    assert bh["measured"] == 1 and bh["unmeasured"] == 0
