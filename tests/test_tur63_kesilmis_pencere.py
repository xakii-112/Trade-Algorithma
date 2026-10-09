"""Tur-63 — İSTENEN pencere ile DOĞAN pencere aynı şey değildir.

Tur-52 bu depoya `veri_kapsami` ön-koşulunu koydu: veri setinin kendi [min,max] aralığında
BOŞ ay varsa hüküm verilmez. Kapı doğruydu; **baktığı yer** eksikti. Delik ARAMAK, ucu
KESİLMİŞ bir pencereyi göremez:

    Hiç doğmamış bir yılın veri setinde İZİ YOKTUR — o yıl bir "delik" değil,
    aralığın kendisinin DIŞIDIR. Kapı deliksiz bir 21 ay görüp PASS der,
    verdikt "5,5 yıl" diye okunur.

Tur-54 bunu ÖLÇTÜ (BULGU-54-1: opsiyon yüzeyi 2021-01'den istendi, 2024-10-14'te doğdu)
ve kapıyı BİLEREK değiştirmedi: aday FAIL ettikten SONRA kapı sıkıştırmak, sonuca göre
kural değiştirmektir. Adayı olmayan bu turda makineye çevrildi (Tur-24/27/31/50 deseni).

ÖLÇÜLDÜ (7 veri seti, istenen pencere ↔ doğan pencere):

    optskew_v1      2021-01-01 istendi → 2024-10-14 doğdu   baş kesme %68,0   ← FAIL
    seasonality_v1  2021-01-01 istendi → 2023-02-01 doğdu   ISINMA beyanıyla %2,4
    babmacro_v1     2021-01-01 istendi → 2021-08-23 doğdu   ISINMA beyanıyla %0,6
    shortflow_v1    2021-01-01 istendi → 2021-03-31 doğdu   ISINMA beyanıyla %0,3
    vrp/insider/rebalance — kurucu pencereyi BİLDİRMİYOR → "ÖLÇÜLEMEDİ" (sessiz PASS DEĞİL)

Üç ayrı durum üç ayrı cümledir (Tur-60/62 deseni): kesilme YOK · kesilme VAR · kesilme
ÖLÇÜLEMEDİ. Üçünü tek bir "PASS"e yıkmak, ölçüm yokluğunu ihlal yokluğu gibi sunardı.

Bu dosya dört şeyi kilitler:
  (A) kesilmiş uç ön-koşulu DÜŞÜRÜR ve verdikt "kenar yok" DEĞİL "hüküm yok" olur,
  (B) bildirilmemiş/geçersiz beyan sessizce PASS'e dönüşemez,
  (C) GEÇEN kapsam ölçümü verdiktte ve DEFTERDE durur (eskiden yalnız FAIL'de görünürdü),
  (D) `run_candidate`'ın her çağıranı pencereyi BEYAN eder (AST — yorum kapıyı tatmin edemez).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.edge_lab import harness, ledger
from research.edge_lab.harness import run_candidate

_KOK = Path(__file__).resolve().parents[1]
_EDGE_LAB = _KOK / "research" / "edge_lab"
_HARNESS_PY = _EDGE_LAB / "harness.py"
_BACKLOG = _EDGE_LAB / "BACKLOG.md"


def _veri(bas: str, son: str, n: int = 200) -> pd.DataFrame:
    """[bas, son] aralığına EŞİT dağılmış, deliksiz sentetik veri seti."""
    rng = np.random.default_rng(0)
    ts = pd.date_range(bas, son, periods=n)
    return pd.DataFrame({
        "ts": ts, "symbol": [f"SYM{i % 10}" for i in range(n)],
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
        "label": rng.integers(0, 2, size=n),
    })


# ---------------------------------------------------------------------------
# KAPI A — kesilmiş uç ön-koşulu DÜŞÜRÜR
# ---------------------------------------------------------------------------
def test_bas_kesilmesi_on_kosulu_dusurur():
    """BULGU-54-1'in kendisi: 5,5 yıl istendi, son 1,7 yılı doğdu, aralığın İÇİ deliksiz."""
    g = harness._stage_coverage(_veri("2024-10-14", "2026-06-23"), "ts",
                                expected_start="2021-01-01", expected_end="2026-06-30",
                                warmup_days=56)
    assert g.passed is False, "3,7 yılı hiç doğmamış veri seti ön-koşulu geçemez"
    assert g.metrics["head_trunc_days"] > 1000
    assert g.metrics["head_frac"] > 0.5
    assert "baş kesme" in g.detail


def test_son_kesilmesi_de_dusurur():
    """Kesilme SON uçta da olabilir: sağlayıcı 2 yıl önce susmuşsa hüküm bugünü kapsamaz."""
    g = harness._stage_coverage(_veri("2021-03-01", "2024-06-30"), "ts",
                                expected_start="2021-01-01", expected_end="2026-06-30",
                                warmup_days=56)
    assert g.passed is False
    assert g.metrics["tail_trunc_days"] > 700
    assert g.metrics["head_frac"] <= harness._MAX_TRUNCATION_FRAC, \
        "baş uç tolerans içinde — kapıyı düşüren şey SON uç olmalı"


def test_tolerans_icindeki_kesilme_gecer():
    """Yayın gecikmesi/tatil payı kapıyı düşürmemeli — kapı gürültüye değil YAPIYA bakar."""
    g = harness._stage_coverage(_veri("2021-03-05", "2026-06-20"), "ts",
                                expected_start="2021-01-01", expected_end="2026-06-30",
                                warmup_days=56)
    assert g.passed is True
    assert 0 < g.metrics["head_frac"] <= harness._MAX_TRUNCATION_FRAC


def test_isinma_beyani_etkin_baslangici_kaydirir():
    """AYNI veri, AYNI istenen pencere: beyan edilen ısınma olmadan FAIL, beyanla PASS.

    seasonality_v1 tam bu durumdadır — aile yapısı gereği pencere içinde 2 yıllık aynı-ay
    geçmişi ister, yani 2021-2022'yi CEVAPLAYAMAZ. Bu bir veri arızası değil, ailenin
    kapsamının BEYANIDIR; beyan edilmediğinde kapı haklı olarak FAIL eder."""
    beyansiz = harness._stage_coverage(_veri("2023-02-01", "2026-05-01"), "ts",
                                       expected_start="2021-01-01", expected_end="2026-06-30",
                                       warmup_days=0)
    beyanli = harness._stage_coverage(_veri("2023-02-01", "2026-05-01"), "ts",
                                      expected_start="2021-01-01", expected_end="2026-06-30",
                                      warmup_days=730)
    assert beyansiz.passed is False, "ısınma beyan edilmeden 2 yıllık kayma kabul edilemez"
    assert beyanli.passed is True
    assert beyanli.metrics["effective_start"] == "2023-01-01"


def test_kesilmis_veri_KENAR_YOK_demez_HUKUM_YOK_der():
    """En önemli ayrım: kesilmiş veri 'bu ailede kenar yok' KANITI değildir.

    `veri_kapsami` bir ön-koşuldur; FAIL ederse beceri kapıları hiç koşmaz ve verdikt
    PRECONDITION_VERDICT olur. FAIL_VERDICT ('kenar YOK') dönseydi, ölçülmemiş bir dönem
    çürütülmüş sayılırdı — defter de o aileyi denenmiş sayıp sonraki adayın DSR eşiğini
    haksız yere yükseltirdi."""
    assert "veri_kapsami" in harness._PRECONDITION_GATES


def test_kesilmis_veri_seti_run_candidate_uzerinden_hukumsuz(tmp_path):
    p = tmp_path / "kesik.csv"
    _veri("2024-10-14", "2026-06-23", n=400).to_csv(p, index=False)
    v = run_candidate("kesik_test", p, ["f1", "f2"], "label",
                      expected_start="2021-01-01", expected_end="2026-06-30", warmup_days=56,
                      ledger_path=tmp_path / "defter.jsonl")
    assert v["verdict"] == harness.PRECONDITION_VERDICT
    assert v["verdict"] != harness.FAIL_VERDICT, "kesilmiş veri 'kenar YOK' KANITI değildir"
    assert [g["name"] for g in v["gates"]] == ["veri_kapsami"], "beceri kapıları KOŞMAMALI"
    assert not (tmp_path / "defter.jsonl").exists(), "kabul edilmeyen veri deftere yazılmaz"


# ---------------------------------------------------------------------------
# KAPI B — "ölçülemedi" ile "ihlal yok" AYRI cümledir
# ---------------------------------------------------------------------------
def test_beyan_yoksa_olculemedi_YAZILIR():
    """Sessiz PASS, bu turun kapattığı kusuru kapının içinde yeniden üretirdi."""
    g = harness._stage_coverage(_veri("2024-10-14", "2026-06-23"), "ts")
    assert g.passed is True, "beyan yokluğu bir İHLAL değildir — kapı uydurmaz"
    assert g.metrics["expected_declared"] is False
    assert "ÖLÇÜLEMEDİ" in g.detail and "BİLDİRİLMEDİ" in g.detail


def test_yarim_beyan_da_olculemedi_sayilir():
    """Yalnız başlangıç bildirilirse pencere yoktur; yarım beyan tam beyan sayılamaz."""
    g = harness._stage_coverage(_veri("2024-10-14", "2026-06-23"), "ts",
                                expected_start="2021-01-01")
    assert g.metrics["expected_declared"] is False
    assert "ÖLÇÜLEMEDİ" in g.detail


def test_gecersiz_beyan_sessizce_PASS_olmaz():
    """Isınma bitişi aşarsa ölçüm tanımsızdır: 0 kesilme raporlamak yanlış beyanı DOĞRU
    ölçüm gibi gösterirdi (sıfıra bölme yerine dürüst FAIL)."""
    g = harness._stage_coverage(_veri("2024-10-14", "2026-06-23"), "ts",
                                expected_start="2021-01-01", expected_end="2021-06-30",
                                warmup_days=5000)
    assert g.passed is False
    assert "GEÇERSİZ" in g.detail


def test_ic_delik_kapisi_hala_calisiyor():
    """Tur-52'nin kanadı kırılmadı: ortadaki delik, uçlar tam olsa bile düşürür."""
    df = pd.concat([_veri("2021-01-01", "2021-06-30", 100),
                    _veri("2023-01-01", "2026-06-30", 100)], ignore_index=True)
    g = harness._stage_coverage(df, "ts", expected_start="2021-01-01",
                                expected_end="2026-06-30", warmup_days=0)
    assert g.passed is False
    assert g.metrics["n_empty"] > 12 and "ay BOŞ" in g.detail


def test_tolerans_sabitleri_kapiyi_olduremez():
    """Bir sonraki tur toleransı 'geçici olarak' gevşetirse kapı ölür — sınır kayıtlı."""
    assert 0.0 < harness._MAX_TRUNCATION_FRAC <= 0.10
    assert 0.0 < harness._MAX_EMPTY_MONTH_FRAC <= 0.05


# ---------------------------------------------------------------------------
# KAPI C — GEÇEN ölçüm de verdiktte ve DEFTERDE durur (BULGU-63-2)
# ---------------------------------------------------------------------------
def test_gecen_kapsam_kapisi_verdiktte_gorunur(tmp_path):
    """Eskiden kapsam kapısı YALNIZ FAIL ederse gates'e giriyordu: hüküm hangi dönemi
    kapsıyor sorusunun yanıtı ne kurucunun çıktısında ne defterde vardı."""
    p = tmp_path / "ok.csv"
    _veri("2021-03-05", "2026-06-20", n=400).to_csv(p, index=False)
    v = run_candidate("kapsam_gorunur", p, ["f1", "f2"], "label",
                      expected_start="2021-01-01", expected_end="2026-06-30", warmup_days=56,
                      ledger_path=tmp_path / "d.jsonl")
    kapsam = next(g for g in v["gates"] if g["name"] == "veri_kapsami")
    assert kapsam["passed"] is True
    assert kapsam["metrics"]["data_start"] == "2021-03-05"
    assert kapsam["metrics"]["data_end"] == "2026-06-20"


def test_defter_satiri_kapsanan_donemi_tasir(tmp_path):
    d = tmp_path / "d.jsonl"
    p = tmp_path / "ok.csv"
    _veri("2021-03-05", "2026-06-20", n=400).to_csv(p, index=False)
    run_candidate("kapsam_defter", p, ["f1", "f2"], "label",
                  expected_start="2021-01-01", expected_end="2026-06-30", warmup_days=56,
                  ledger_path=d)
    satir = json.loads(d.read_text(encoding="utf-8").splitlines()[-1])
    assert satir["kapsam"]["data_start"] == "2021-03-05"
    assert satir["kapsam"]["expected_start"] == "2021-01-01"
    assert satir["kapsam"]["expected_declared"] is True


def test_eski_defter_satirlari_kapsamsiz_okunabilir(tmp_path):
    """Defter yalnız-eklemedir: geçmiş satırlar bu alanı TAŞIMAZ ve geriye doldurulmaz —
    okuyucu bunu çökmeden geçebilmeli (aksi halde yeni alan eski kaydı okunamaz kılardı)."""
    d = tmp_path / "d.jsonl"
    d.write_text(json.dumps({"ts": "2026-01-01T00:00:00+00:00", "candidate": "eski",
                             "all_passed": False}, ensure_ascii=False) + "\n", encoding="utf-8")
    satirlar = ledger.read_entries(d)
    assert satirlar[0].get("kapsam") is None
    assert ledger.n_trials_for("yeni", 4, d) == 8
    assert "eski" in ledger.summary(d)


# ---------------------------------------------------------------------------
# KAPI D — her `run_candidate` çağıranı pencereyi BEYAN eder (AST)
# ---------------------------------------------------------------------------
def _run_candidate_cagrilari(p: Path) -> list[ast.Call]:
    agac = ast.parse(p.read_text(encoding="utf-8"))
    return [d for d in ast.walk(agac)
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
            and d.func.id == "run_candidate"]


def test_her_kurucu_pencereyi_beyan_eder():
    """Yeni bir kurucu pencereyi bildirmeyi UNUTURSA kapı burada düşer.

    Tarama AST üzerindedir: bu dosyanın ve raporun metni `expected_start` ifadesini
    defalarca geçiriyor; ham metinde arayan bir kapı kendi anlatısına takılırdı (Tur-49)."""
    bulunan = 0
    for py in sorted(_EDGE_LAB.glob("build_*.py")):
        for cagri in _run_candidate_cagrilari(py):
            bulunan += 1
            anahtarlar = {k.arg for k in cagri.keywords}
            eksik = {"expected_start", "expected_end", "warmup_days"} - anahtarlar
            assert not eksik, f"{py.name}: run_candidate çağrısında eksik beyan: {sorted(eksik)}"
    assert bulunan >= 4, f"kurucu taraması boşa düştü (bulunan çağrı: {bulunan})"


def test_kapilardan_gecir_pencereyi_main_den_alir():
    """Beyan sabit değil, kurucunun GERÇEK argümanından gelmeli: `--since/--until` ile
    dar bir pencere istenirse kapı o pencereyi ölçmeli, kodda donmuş bir tarihi değil."""
    for py in sorted(_EDGE_LAB.glob("build_*.py")):
        for cagri in _run_candidate_cagrilari(py):
            for k in cagri.keywords:
                if k.arg in ("expected_start", "expected_end"):
                    assert isinstance(k.value, ast.Name), \
                        f"{py.name}: {k.arg} sabit yazılmış — kurucunun argümanı olmalı"


def test_stage_coverage_pencere_parametrelerini_KABUL_eder():
    """Regresyon kapısı: kapı eski (yalnız iç delik) imzasına geri dönemez."""
    agac = ast.parse(_HARNESS_PY.read_text(encoding="utf-8"))
    fn = next(d for d in ast.walk(agac)
              if isinstance(d, ast.FunctionDef) and d.name == "_stage_coverage")
    kwonly = {a.arg for a in fn.args.kwonlyargs}
    assert {"expected_start", "expected_end", "warmup_days"} <= kwonly


def test_backlog_artik_kapiyi_KOR_diye_belgelemiyor():
    """Belge↔kod kilidi (Tur-61 deseni): BACKLOG'un açık kalem satırı kapının bu durumu
    'göremediğini' yazıyordu. Kapı artık görüyor; belge onu kör ilan etmeye devam ederse
    ikisinden biri yalan söylüyordur."""
    metin = _BACKLOG.read_text(encoding="utf-8")
    assert "durumunu göremiyor" not in metin, \
        "BACKLOG hâlâ kapıyı kör ilan ediyor — belge kodla çelişiyor"
    assert "AÇIK KALEM KAPANDI (Tur-63)" in metin, "kapanış BACKLOG'a yazılmalı"


# ---------------------------------------------------------------------------
# KAPI E — aynı kusurun KAYNAKTAKİ kardeşi: yazılıp okunamayan kırpılma bayrağı
# ---------------------------------------------------------------------------
def test_get_all_kirpik_serileri_cerceveyle_BIRLIKTE_tasir(monkeypatch):
    """`get()` seri başına `attrs["kirpildi"]` yazıyordu; Series → DataFrame geçişinde
    attrs KAYBOLUYOR, yani bayrak hiçbir çağırana ulaşmıyordu (yazıldı, okunamadı)."""
    from nihai.data import fred_data

    def sahte_get(seri, *, since="2015-01-01", **kw):
        s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), name=seri)
        s.attrs["kirpildi"] = (seri == "KISA")
        return s

    monkeypatch.setattr(fred_data, "get", sahte_get)
    df = fred_data.get_all(["UZUN", "KISA"], since="2015-01-01")
    assert df.attrs["kirpik"] == ["KISA"], "kırpık seri adı çerçeveyle taşınmalı"


def test_babmacro_kirpilmayi_KAYNAKTAN_okur(monkeypatch):
    """Ayırt edici: çerçevenin VERİSİ kırpık görünmüyor (ilk gözlem `since` ile aynı gün),
    yalnız attrs kırpık diyor. Kurucu ölçümü elle yeniden yapsaydı uyarı ÇIKMAZDI."""
    from loguru import logger

    from research.edge_lab import build_babmacro_dataset as bm

    gunler = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=30))
    ham = pd.DataFrame({"T10Y2Y": [1.0], "NFCI": [-0.5]}, index=[pd.Timestamp("2021-01-04")])
    ham.attrs["kirpik"] = ["NFCI"]
    monkeypatch.setattr(bm.fred_data, "get_all", lambda *a, **k: ham)
    kayit: list[str] = []                       # loguru → caplog'a düşmez, kendi sink'imiz
    sink = logger.add(lambda m: kayit.append(str(m)), level="WARNING")
    try:
        bm.makro_gunluk(gunler, "2021-01-04")
    finally:
        logger.remove(sink)
    assert any("KIRPIK" in k and "NFCI" in k for k in kayit), \
        "kırpılma bayrağı kaynaktan okunmuyor"


@pytest.mark.parametrize("modul,beklenen", [
    ("build_babmacro_dataset", 224), ("build_optskew_dataset", 56),
    ("build_seasonality_dataset", 730), ("build_shortflow_dataset", 84),
])
def test_isinma_beyanlari_olculmus_degerlerde_sabit(modul, beklenen):
    """Beyanlar ÖLÇÜMDEN geldi (rapor §2). Bir sonraki tur birini 'veri geçsin diye'
    büyütürse, kapı gevşemiş olur — değişiklik bu testten geçmek zorundadır."""
    import importlib
    m = importlib.import_module(f"research.edge_lab.{modul}")
    assert m.ISINMA_GUN == beklenen
