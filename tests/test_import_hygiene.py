"""Import hijyeni sözleşmeleri (Tur-26) — ağsız, alt-süreçli, deterministik.

İki şeyi KİLİTLER:

(1) **`import nihai.ai` UCUZ kalır.** `nihai/ai/__init__.py`'nin kendi gerekçesi bunu
    zaten söylüyordu ama kod eager import ediyordu: `decision → quant_math →
    sklearn.covariance + scipy.stats` zinciri ölçümde **3854 ms** tutuyordu ve bunu
    `from nihai.ai import registry` diyen CANLI yol (`execution/neural_shadow`) ile her
    `web/api_extra` AI ucu ödüyordu. Tüm alt modüller lazy'ye çevrilince **11 ms**.
    Bu test o kazancın sessizce geri gelmesini engeller — biri `__init__`'e eager import
    eklerse kırılır.

(2) **Opsiyonel RESEARCH bağımlılığı tüm kalite kapısını düşüremez.** `research/pairs_test.py`
    statsmodels'i modül düzeyinde KATI import ediyordu; `tests/test_pairs_math.py` onu
    import ettiği için statsmodels'siz bir ortamda pytest TOPLAMA fazında duruyor ve
    **2178 testin tamamı** koşmuyordu (gözlem: "Interrupted: 1 error during collection").

Alt süreç kullanılır: bu testler `sys.modules`'ın TEMİZ olmasını gerektirir, oysa tam
suite koşusunda sklearn/scipy zaten yüklüdür (başka testler yükler)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _kos(kod: str) -> subprocess.CompletedProcess[str]:
    """Kodu temiz bir alt süreçte koştur (repo kökü cwd — kök conftest sys.path'i kurar)."""
    return subprocess.run(
        [sys.executable, "-c", kod], cwd=str(ROOT),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )


# ---------------------------------------------------------------------------
# (1) nihai.ai lazy sözleşmesi
# ---------------------------------------------------------------------------
def test_import_nihai_ai_agir_bagimlilik_cekmez():
    """`import nihai.ai` sklearn/scipy/pandas/torch/statsmodels'i sys.modules'a SOKMAMALI."""
    r = _kos(
        "import sys; import nihai.ai; "
        "print(sorted(m for m in ('sklearn','scipy','pandas','torch','statsmodels','matplotlib') "
        "if m in sys.modules))"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", (
        f"`import nihai.ai` ağır bağımlılık çekti: {r.stdout.strip()}\n"
        "nihai/ai/__init__.py'ye eager import eklenmiş olmalı — alt modüller LAZY kalmalı "
        "(PEP 562 __getattr__). Gerekçe ve ölçüm dosyanın başındaki blokta."
    )


def test_import_nihai_ai_alt_modul_yuklemez():
    """Yan etki yok: hiçbir `nihai.ai.*` alt modülü kendiliğinden yüklenmemeli."""
    r = _kos("import sys; import nihai.ai; "
             "print(sorted(m for m in sys.modules if m.startswith('nihai.ai.')))")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", f"eager yüklenen alt modül(ler): {r.stdout.strip()}"


def test_all_isimlerinin_hepsi_oznitelik_olarak_cozulur():
    """Lazy'lik API'yi BOZMAMALI: `import nihai.ai` + `nihai.ai.X` her __all__ adı için
    çalışmalı (eager import'un sağladığı tek şey buydu; PEP 562 kancası devralıyor)."""
    r = _kos(
        "import nihai.ai as A; "
        "eksik=[n for n in A.__all__ if not hasattr(A, n)]; "
        "print('EKSIK=' + repr(eksik))"
    )
    assert r.returncode == 0, r.stderr
    assert "EKSIK=[]" in r.stdout, r.stdout


def test_bilinmeyen_oznitelik_attributeerror_verir():
    """Kanca her adı yutmamalı — yazım hatası sessizce ImportError'a dönüşmesin."""
    r = _kos(
        "import nihai.ai as A\n"
        "try:\n"
        "    A.boyle_bir_modul_yok\n"
        "except AttributeError:\n"
        "    print('OK')\n"
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# (2) research.pairs_test — statsmodels YOKKEN de import edilebilmeli
# ---------------------------------------------------------------------------
def test_pairs_test_statsmodels_yokken_import_edilebilir():
    """statsmodels sys.modules'ta bloklanmışken bile modül import edilebilmeli —
    aksi hâlde TEK opsiyonel bağımlılık tüm test toplamasını düşürür."""
    r = _kos(
        "import sys\n"
        "for m in list(sys.modules):\n"
        "    if m.startswith('statsmodels'):\n"
        "        del sys.modules[m]\n"
        "sys.modules['statsmodels'] = None\n"
        "sys.modules['statsmodels.tsa'] = None\n"
        "sys.modules['statsmodels.tsa.stattools'] = None\n"
        "import research.pairs_test as pt\n"
        "print('IMPORT_OK', callable(pt.coint), callable(pt.simulate_pair))\n"
    )
    assert r.returncode == 0, r.stderr
    assert "IMPORT_OK True True" in r.stdout, r.stdout


def test_pairs_test_coint_statsmodels_yoksa_yonergeli_hata_verir():
    """Yer tutucu sessizce yanlış sonuç ÜRETMEMELİ: gerçekten çağrılırsa kurulum
    yönergesiyle ModuleNotFoundError yükseltmeli."""
    r = _kos(
        "import sys\n"
        "for m in list(sys.modules):\n"
        "    if m.startswith('statsmodels'):\n"
        "        del sys.modules[m]\n"
        "sys.modules['statsmodels'] = None\n"
        "sys.modules['statsmodels.tsa'] = None\n"
        "sys.modules['statsmodels.tsa.stattools'] = None\n"
        "import research.pairs_test as pt\n"
        "try:\n"
        "    pt.coint([1.0, 2.0], [1.0, 2.0])\n"
        "except ModuleNotFoundError as e:\n"
        "    print('RAISED', 'pip install' in str(e))\n"
    )
    assert r.returncode == 0, r.stderr
    assert "RAISED True" in r.stdout, r.stdout
