"""Tur-65 — dış optimizasyon kataloğundan UYGULANAN maddelerin kilidi.

Her biri "bir daha sessizce kapanmasın" diye kilitlenir:

1. **#27 — paylaşılan HTTP istemcisi.** `httpx.get/post` modül kısayolları HER ÇAĞRIDA yeni
   bir `Client` kurar; bedel ağ değil, SSL bağlamı + CA demetidir. ÖLÇÜLDÜ (ağsız, 15
   yineleme): `httpx.Client()` **224,66 ms**, `requests.Session()` **0,02 ms** — vergi
   httpx'e ÖZGÜ. Turun İLK geçişi düzeltmeyi yalnız `telegram_bot`a uygulamıştı; kardeş
   çağıranlar (`nasdaq_calendar`, `screen`) açıktı ve `nasdaq_calendar` DÖNGÜ içindeydi
   (`fetch_next_earnings` gün gün ilerler → 14 günlük pencere = ~3,1 sn saf kurulum).
   Artık üçü de `nihai.net.shared_client()` kullanır.
2. **#26 — HMM uyum önbelleği.** `/api/quant/regime` ısınmış p50 **2334,8 ms** ölçüldü
   (panonun 55 GET ucunun toplamının %75'i); profil `gaussian_hmm_fit`i gösterdi (k=2,3,4
   için üç EM, fit başına ~152 adım). Fonksiyon SAF+DETERMİNİSTİK olduğu için memoize
   edildi → uç **147,1 ms** (**15,9×**), pano toplamı 3085,7 → 892,2 ms (**3,46×**).
3. **#2 — pyarrow artık BEYANLI.** Tur-64 parquet yan dosyasını ölçtü ama ölçüm BASE
   yorumlayıcıdaydı; üretimin koştuğu `.venv`de paket YOKTU → hızlı yol üretimde hiç
   çalışmadı, Tur-64 testleri de `importorskip` ile sessizce atlandı (Tur-26'nın "kalite
   kapısı kördü" dersinin birebir tekrarı).
4. **#19 — GPU opt-in KALIR.** CUDA ölçüldü (quant_transformer 135,21 → 1,81 sn = 74,7×)
   ama `--device` varsayılanı bilerek `cpu`dur: GPU/bf16 farklı kayan-nokta sırasıdır,
   geçmiş koşularla bit-denk DEĞİLDİR (`--optimizer adafactor` kararının aynısı).
"""
from __future__ import annotations

import ast
import pathlib

import httpx
import numpy as np
import pytest

from nihai import net
from nihai.ai import regime_hmm as rh

KOK = pathlib.Path(__file__).resolve().parents[1]

# #27'nin kapsadığı üç çağıran — biri eklenip unutulursa test değil, KAPSAM eksilir.
HTTPX_CAGIRANLARI = (
    KOK / "nihai" / "notifier" / "telegram_bot.py",
    KOK / "nihai" / "data" / "nasdaq_calendar.py",
    KOK / "nihai" / "execution" / "screen.py",
)


@pytest.fixture
def temiz_istemci():
    """Her test paylaşılan istemciyi kendi başına kurup söksün (süreç sızdırmasın)."""
    net.close_shared_client()
    yield
    net.close_shared_client()


# --- #27: paylaşılan httpx.Client -------------------------------------------------
def test_http_istemcisi_tembel_kurulur(temiz_istemci):
    """`import nihai.net` TEK BAŞINA istemci kurmamalı — SSL bağlamı 225 ms'dir."""
    assert net._client is None
    assert isinstance(net.shared_client(), httpx.Client)


def test_http_istemcisi_tekrar_kullanilir(temiz_istemci):
    assert net.shared_client() is net.shared_client(), (
        "her çağrıda yeni Client = 224,66 ms/çağrı vergisi geri geldi")


def test_kapatinca_sifirlanir(temiz_istemci):
    ilk = net.shared_client()
    net.close_shared_client()
    assert net._client is None
    assert net.shared_client() is not ilk


def test_telegram_eski_adi_ayni_istemciyi_verir(temiz_istemci):
    """`telegram_bot._http` geriye dönük ad; kendi istemcisini kurmamalı."""
    import nihai.notifier.telegram_bot as tb

    assert tb._http() is net.shared_client()


@pytest.mark.parametrize("yol", HTTPX_CAGIRANLARI, ids=lambda p: p.name)
def test_hicbir_cagiran_modul_duzeyi_httpx_kullanmaz(yol):
    """`httpx.get(...)` / `httpx.post(...)` ÇAĞRISI kalmamalı (import durabilir).

    `ast` ile denetlenir; grep değil — yorum satırındaki bir örnek testi düşürmesin.
    """
    kalanlar = []
    for d in ast.walk(ast.parse(yol.read_text(encoding="utf-8"))):
        if not isinstance(d, ast.Call):
            continue
        f = d.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "httpx" and f.attr in ("get", "post", "put", "delete")):
            kalanlar.append(f"{f.value.id}.{f.attr} (satır {d.lineno})")
    assert not kalanlar, f"{yol.name}: modül düzeyi httpx çağrısı kaldı: {kalanlar}"


def _kwargler(yol: pathlib.Path) -> dict[str, set[str]]:
    """fonksiyon adı → içindeki çağrıların anahtar-kelime argümanları ('ad=ifade')."""
    bulunan: dict[str, set[str]] = {}
    for fn in ast.walk(ast.parse(yol.read_text(encoding="utf-8"))):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for d in ast.walk(fn):
            if isinstance(d, ast.Call):
                for kw in d.keywords:
                    if kw.arg:
                        bulunan.setdefault(fn.name, set()).add(f"{kw.arg}={ast.unparse(kw.value)}")
    return bulunan


def test_cagri_basina_timeout_ve_yonlendirme_korundu():
    """Paylaşılan istemci ORTAK varsayılan DAYATMAMALI: `sendPhoto` 20 sn, `getUpdates`
    long-poll `timeout + 10`, Wikipedia `follow_redirects=True` ister. httpx'te ikisi de
    İSTEK BAŞINA parametredir; istemciye taşımak bu farkları sessizce ezerdi."""
    tg = _kwargler(HTTPX_CAGIRANLARI[0])
    assert "timeout=20" in tg.get("_post_photo", set())
    assert "timeout=timeout + 10" in tg.get("_get_updates", set())

    nas = _kwargler(HTTPX_CAGIRANLARI[1])
    assert "timeout=timeout" in nas.get("_http_fetch_date", set())

    scr = _kwargler(HTTPX_CAGIRANLARI[2])
    assert {"timeout=20", "follow_redirects=True"} <= scr.get("_read_html_tables", set())


# --- #26: HMM uyum önbelleği ------------------------------------------------------
def _seri(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 0.004, n // 2), rng.normal(0.0, 0.012, n - n // 2)])


def test_uyum_onbellegi_isabet_eder():
    x = _seri()
    rh._hmm_fit_cached.cache_clear()
    a = rh.gaussian_hmm_fit(x, 2)
    onceki = rh._hmm_fit_cached.cache_info()
    b = rh.gaussian_hmm_fit(x, 2)
    sonraki = rh._hmm_fit_cached.cache_info()
    assert a is not None and a == b, "önbellek farklı sonuç verdi — memoization SAF değil"
    assert sonraki.hits == onceki.hits + 1, "ikinci çağrı EM'i yeniden koşturdu (2,3 sn)"


def test_tek_bar_degisince_onbellek_isabet_etmez():
    """Anahtar serinin BAYTLARIdır: bayat bir model dönemez."""
    x = _seri()
    rh._hmm_fit_cached.cache_clear()
    rh.gaussian_hmm_fit(x, 2)
    y = x.copy()
    y[-1] += 1e-9
    onceki = rh._hmm_fit_cached.cache_info()
    rh.gaussian_hmm_fit(y, 2)
    assert rh._hmm_fit_cached.cache_info().misses == onceki.misses + 1


def test_donen_sozluk_degistirilse_de_onbellek_kirlenmez():
    """Çağıran sonucu değiştirebilir; önbellekteki nesne PAYLAŞILMAMALI."""
    x = _seri()
    rh._hmm_fit_cached.cache_clear()
    a = rh.gaussian_hmm_fit(x, 2)
    assert a is not None
    a["means"][0] = 12345.0
    a["transition"][0][0] = 0.5
    b = rh.gaussian_hmm_fit(x, 2)
    assert b is not None
    assert b["means"][0] != 12345.0 and b["transition"][0][0] != 0.5


def test_onbellek_sinirli():
    """Sınırsız önbellek, uzun koşan bir araştırma betiğinde bellek sızıntısıdır."""
    assert rh._hmm_fit_cached.cache_info().maxsize == rh._FIT_CACHE_MAX
    assert 1 <= rh._FIT_CACHE_MAX <= 64


def test_gecersiz_girdi_onbellege_girmez():
    """Doğrulama SARMALAYICIDA kalmalı: geçersiz girdi EM'e (ve önbelleğe) hiç ulaşmamalı.

    Önemi kozmetik değil: doğrulama önbelleklenen gövdeye kaysaydı, her bozuk girdi bir
    önbellek girişi harcar ve 8 girişlik LRU'yu gerçek uyumlardan boşaltırdı."""
    rh._hmm_fit_cached.cache_clear()
    assert rh.gaussian_hmm_fit(_seri(20), 2) is None                    # 30 gözlemden kısa
    assert rh.gaussian_hmm_fit(_seri(100), 99) is None                  # n_states ∉ [2,6]
    assert rh.gaussian_hmm_fit(_seri(50), 6) is None                    # 10·k'dan kısa
    bozuk = _seri(200)
    bozuk[7] = np.nan
    assert rh.gaussian_hmm_fit(bozuk, 2) is None                        # NaN
    assert rh._hmm_fit_cached.cache_info().currsize == 0


# --- #2: pyarrow beyanı -----------------------------------------------------------
def test_pyarrow_requirements_te_beyanli():
    """Yorum satırı DEĞİL, gerçek bir gereksinim satırı olmalı."""
    satirlar = (KOK / "requirements.txt").read_text(encoding="utf-8").splitlines()
    beyan = [s.strip() for s in satirlar
             if s.strip().startswith("pyarrow") and not s.strip().startswith("#")]
    assert beyan, ("pyarrow yorumda kaldı — Tur-64'ün 50,4× parquet yolu üretim "
                   "venv'inde yine sessizce ölür (Tur-26 dersi)")


# --- #19: GPU opt-in kalır --------------------------------------------------------
def test_train_neural_device_varsayilani_cpu_kalir():
    """74,7× ölçüldü diye varsayılanı çevirmek geçmiş koşuları bit-denk olmaktan
    çıkarır. GPU bilinçli bir SEÇİM olarak kalmalı."""
    kaynak = (KOK / "research" / "train_neural.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--device", default="cpu"' in kaynak


def test_train_neural_cuda_ipucu_var():
    """Ölçülen düğme GÖRÜNÜR olmalı: CUDA varken cpu seçilirse operatöre söylenir
    ("belgesiz düğme çevrilmeyen düğmedir" — Tur-61)."""
    kaynak = (KOK / "research" / "train_neural.py").read_text(encoding="utf-8")
    assert "cuda.is_available()" in kaynak
    assert "[ipucu]" in kaynak
