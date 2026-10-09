"""
lab.py — KOD YAZMA/ÇALIŞTIRMA LABORATUVARI (Tur-16 WP-C, docs/AI_V3_DERIN_BEYIN_SPEC.md §3.9
taslağının bu görev talimatına göre genişletilmiş hali — bkz. teslim raporu "bilinçli
sapmalar": burada ana API adı `run_sandbox` [spec taslağı `run_code` diyordu]). KARAR-DIŞI.

⚙️ KULLANIM KİPİ: **MANUEL / CLI-only.** Bu modülü üretim akışında (tarayıcı, sinyal, kayıtçı,
web) çağıran YOKTUR ve bu KASITLIDIR — `brain.py` yalnız ÖNERİ metninde adını anar. Kod
üreten/çalıştıran bir bileşenin canlı döngüye kendiliğinden girmesi istenmez; operatör
bilinçli çağırır. Denetim (2026-07-26 B-8) "ölü kod mu?" diye sorduğu için bu satır eklendi:
ölü değil, **çağrılmayı bekleyen bir araç**. Ölü koda dönüştüğüne karar verilirse silinmeli,
sessizce durmamalı.

Ne yapar: AI'ın "soru → Python analiz kodu → çalıştır → yorumla" döngüsünü sağlar.
İki ayrı kullanım biçimi:
  1) `run_sandbox(code_str, inputs=None, timeout_s=30)` — SERBEST metin kodu KISITLI bir
     `exec` ortamında çalıştırır (beyaz-liste import + beyaz-liste builtins + süre limiti
     + stdout yakalama). Genel-amaçlı ama tehlikeli (rastgele kod).
  2) `run_experiment(name, **kwargs)` — ÖNCEDEN YAZILMIŞ, GÜVENİLİR (bu modülün kendi
     kodu — sandbox'a hiç girmez), DETERMİNİSTİK deney şablonları. LLM'siz de tam
     değerlidir (plan WP-C: "lab hiç LLM'siz de değerlidir").

GÜVENLİK KAPSAMI — DÜRÜST SINIR (`run_sandbox` için): bu bir KÖTÜ NİYETLİ/düşman koda karşı
process/OS-düzeyi izolasyon SİSTEMİ DEĞİLDİR (aynı Python süreci/iş parçacığında çalışır).
Anayasa bağlamı zaten dahili/KARAR-DIŞI kullanım (LLM'in ürettiği veya kullanıcının yazdığı
ANALİZ kodu — üretim dosyalarını YANLIŞLIKLA/gafletle bozmasın diye bir dikkat-duvarı).
Kesin engellenen SOMUT vektörler (testli):
  - `open`/dosya-yazımı: builtin `open` sandbox'ta TANIMSIZ (NameError).
  - beyaz-liste dışı `import`/`__import__(...)`: yalnız math/numpy/pandas/scipy (+alt-
    modülleri) ve `nihai.ai.quant_math`/`nihai.ai.advanced_math` (LAZY, ImportError'a
    dayanıklı — advanced_math henüz yazılmamışsa sandbox'a hiç eklenmez, `import` denemesi
    normal ModuleNotFoundError verir). Başka HER isim (`os`, `sys`, `subprocess`, `socket`,
    `pathlib`, `ctypes`, ...) `ImportError` ile reddedilir.
  - `exec`/`eval`/`compile`/`input`: beyaz-liste builtins'te YOK → NameError.
  - ağ erişimi: `socket`/`requests`/`urllib` importu beyaz-liste dışı olduğundan mümkün değil.
  - süre aşımı: arka-plan iş parçacığı + `timeout_s` sonrası ZORLA döner (aşağıya bakınız).
BİLİNÇLİ OLARAK KAPSAM DIŞI (test EDİLMEZ, gizlenmez — burada açıkça belgelenir):
  - öznitelik-zinciri gadget'ları (ör. `().__class__.__mro__[1].__subclasses__()` ile
    dolaylı `os` erişimi) — tam engel `getattr`/`type`/`__class__` erişimini de kapatmayı
    gerektirir, bu da şablon kodun (numpy/pandas kullanımının) çoğunu kırar; bilinçli takas.
  - pandas'ın KENDİ dosya G/Ç fonksiyonları (`pd.read_csv`/`to_csv` vb.) — pandas modülü
    TAMAMEN beyaz-listede (aksi halde neredeyse hiçbir analiz kodu çalışmaz); bu fonksiyonlar
    builtin `open` ÜZERİNDEN geçmez, kendi C/uzantı kodlarıyla dosyaya erişir. `data/`
    dizinine böyle bir erişim OS-düzeyinde engellenmez (yalnız kaza/gaflet varsayımı).
  - bellek tavanı: `resource.setrlimit` Windows'ta YOKTUR (bu geliştirme ortamı win32) —
    platform-bağımsız bir üst-sınır bu modülde uygulanmadı (görev talimatı da yalnız süre
    limiti istiyordu, bellek tavanından bahsetmiyordu — bilinçli kapsam daraltması).

ZAMAN AŞIMI — CPython SINIRI: saf-Python döngüleri (`while True: ...` gibi, her geri-sıçrama
bytecode'unda kontrol edilir) pratikte kesilir; TEK bir uzun C-düzeyi bloklu çağrı (zaten
`time`/`os` beyaz-listede değil, ama teorik olarak uzun bir numpy/scipy C-çağrısı) CPython'da
ÖNALIMLI olarak sonlandırılamaz. Bu durumda dahi `run_sandbox` `timeout_s` civarında HATA ile
DÖNER (arka-plan iş parçacığı `daemon=True` — süreci asla engellemez); yalnız kod GERÇEKTEN
arka planda bir süre daha CPU tüketebilir. `PyThreadState_SetAsyncExc` ile en-iyi-çaba bir
kesme denemesi de yapılır (çoğu saf-Python döngüsünü gerçekten durdurur).

Loglama: HER `run_sandbox`/`run_experiment` çağrısı `data/ai_lab_log.jsonl`'e ATOMİK eklenir
(kod/deney-adı + çıktı-özeti + ts UTC — denetlenebilirlik; `nihai.atomic_io` ile, OneDrive/AV
kilidine dayanıklı — ACI DERS, bkz. `atomic_io.py` başlığı).

Şablonlar (`run_experiment(name, **kwargs)`, hepsi DETERMİNİSTİK, hiçbiri 'kenar' iddiası
DÖNMEZ — çıktı dict'lerinde her zaman `'not': 'tanımlayıcı analiz; kenar iddiası değildir'`):
  - `vol_model_karsilastirma(closes, opens=None, highs=None, lows=None, window=20)`
  - `korelasyon_rejimi(panel, window=60)`
  - `olay_etudu(closes, event_indices, pre=10, post=10)`
  - `dagilim_testi(returns)`
  - `ozellik_tarama(X, y, feature_names=None, seed=42)`

`ask_llm_analysis(question, context=None)`: `nihai.ai.router.complete("judgment", ...)`
üzerinden FAIL-SOFT (anahtar yoksa/başarısızsa deterministik şablon cevaba düşer — router.py
sözleşmesiyle BİREBİR aynı desen).
"""
from __future__ import annotations

import builtins as _builtins_mod
import contextlib
import io
import json
import math
import threading
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import scipy
from loguru import logger
from scipy import stats

from nihai.ai import quant_math, router
from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

AI_LAB_LOG_FILE = DATA_DIR / "ai_lab_log.jsonl"
_LOCK = threading.Lock()

_CODE_LOG_TRUNC = 4000
_OUTPUT_LOG_TRUNC = 500

_NOT_KENAR_IDDIASI_DEGIL = "tanımlayıcı analiz; kenar iddiası değildir"


def _utc_now_iso() -> str:
    """UTC ts (tz-aware, saniye çözünürlüğü) — router.py/registry.py ile aynı konvansiyon."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Sandbox — beyaz-liste import + builtins
# ---------------------------------------------------------------------------
def _lazy_advanced_math() -> Any:
    """`nihai.ai.advanced_math`'i dener (Dalga-2 paralel teslim — bu depoda henüz YOK).
    Yoksa None (fail-soft; sandbox globals'ına hiç eklenmez, `import` denemesi kendi
    normal ModuleNotFoundError'ını verir — modül docstring'i)."""
    try:
        from nihai.ai import advanced_math
    except ImportError:
        return None
    return advanced_math


_ALLOWED_TOP: frozenset[str] = frozenset({"math", "numpy", "pandas", "scipy"})
_ALLOWED_DOTTED_PREFIXES: tuple[str, ...] = ("numpy.", "pandas.", "scipy.")
_ALLOWED_NIHAI_LEAVES: frozenset[str] = frozenset({"quant_math", "advanced_math"})
_ALLOWED_NIHAI_EXACT: frozenset[str] = frozenset(
    {f"nihai.ai.{leaf}" for leaf in _ALLOWED_NIHAI_LEAVES})


def _safe_import(name: str, globals: Any = None, locals: Any = None,  # noqa: A002 — __import__ imzası
                 fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    """Sandbox `__import__` sarmalayıcısı: yalnız beyaz-listedeki modülleri gerçekten
    import eder (`builtins.__import__`'a delege), aksi halde `ImportError`. Göreli import
    (`level != 0`) daima reddedilir. `name` ve `fromlist` gerçek import ÇAĞRISINDAN ÖNCE
    kontrol edildiği için beyaz-liste-dışı bir modül sys.modules'a HİÇ girmez (bkz. modül
    docstring'i güvenlik kapsamı)."""
    if level != 0:
        raise ImportError("göreli import (relative import) sandbox'ta yasak")
    fromlist = fromlist or ()
    allowed = (
        name in _ALLOWED_TOP
        or any(name.startswith(p) for p in _ALLOWED_DOTTED_PREFIXES)
        or name in _ALLOWED_NIHAI_EXACT
        or (name == "nihai.ai" and bool(fromlist) and all(f in _ALLOWED_NIHAI_LEAVES for f in fromlist))
    )
    if not allowed:
        raise ImportError(
            f"'{name}' sandbox beyaz-listesinde değil "
            "(izinli: math/numpy/pandas/scipy + nihai.ai.quant_math/advanced_math)")
    return _builtins_mod.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format",
    "frozenset", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "pow", "print", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip", "type",
    "True", "False", "None",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "IndexError",
    "KeyError", "NotImplementedError", "OverflowError", "RuntimeError", "StopIteration",
    "TypeError", "ValueError", "ZeroDivisionError", "FloatingPointError", "OSError",
)
_SAFE_BUILTINS: dict[str, Any] = {
    nm: getattr(_builtins_mod, nm) for nm in _SAFE_BUILTIN_NAMES if hasattr(_builtins_mod, nm)
}
_SAFE_BUILTINS["__import__"] = _safe_import
# NOT: bilinçli olarak YOK: open/exec/eval/compile/input/getattr/setattr/delattr/hasattr/
# globals/locals/vars/dir/__build_class__ — modül docstring'i güvenlik kapsamı notuna bakınız.


def _sandbox_globals(inputs: dict[str, Any] | None) -> dict[str, Any]:
    """Sandbox `exec` globals'ı: kısıtlı builtins + önceden-yüklenmiş modüller (kod bunları
    `import` YAZMADAN da kullanabilir — `np`/`pd`/`math`/`scipy`/`quant_math`) + `inputs`
    sözlüğünün BİREBİR kopyası (çağıranın verdiği veri, örn. `{'closes': arr}`)."""
    g: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    g["math"] = math
    g["np"] = np
    g["numpy"] = np
    g["pd"] = pd
    g["pandas"] = pd
    g["scipy"] = scipy
    g["quant_math"] = quant_math
    adv = _lazy_advanced_math()
    if adv is not None:
        g["advanced_math"] = adv
    for k, v in (inputs or {}).items():
        g[str(k)] = v
    return g


def _best_effort_interrupt(tid: int | None) -> None:
    """CPython C-API üzerinden async istisna enjekte etmeyi DENER (en-iyi-çaba — bkz.
    modül docstring'i ZAMAN AŞIMI notu). Başarısızlık sessizce yutulur; `run_sandbox`'ın
    döndürdüğü zaman-aşımı sonucu bundan BAĞIMSIZDIR (zaten `thread.join` sonrası döner)."""
    if tid is None:
        return
    try:
        import ctypes
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(TimeoutError))
    except Exception as e:  # noqa: BLE001 — en-iyi-çaba temizlik; sonucu etkilemez
        logger.debug(f"[ai.lab] async-interrupt denemesi başarısız: {e}")


def _log_run(kind: str, payload: dict[str, Any]) -> None:
    """`AI_LAB_LOG_FILE`'a ATOMİK ekleme (oku+ekle+tam-yeniden-yaz — portfolio.py/
    registry.py deseni). Hata → sessiz-uyarı (loglama başarısızlığı ana sonucu etkilemez)."""
    row = {"ts": _utc_now_iso(), "kind": kind, **payload}
    line = json.dumps(row, ensure_ascii=False, default=str)
    try:
        with _LOCK:
            AI_LAB_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            prev = AI_LAB_LOG_FILE.read_text(encoding="utf-8") if AI_LAB_LOG_FILE.exists() else ""
            atomic_write_text(AI_LAB_LOG_FILE, prev + line + "\n")
    except OSError as e:
        logger.warning(f"[ai.lab] {AI_LAB_LOG_FILE.name} yazılamadı: {e}")


def run_sandbox(code_str: str, inputs: dict[str, Any] | None = None,
                timeout_s: int = 30) -> dict[str, Any]:
    """`code_str`'i KISITLI `exec` ortamında çalıştırır (modül docstring'i — beyaz-liste
    import/builtins + süre limiti + stdout yakalama). `inputs` sandbox globals'ına
    doğrudan kopyalanır. Sonuç = sandbox'taki `result` adlı değişken (kod onu ATAMAZSA None).

    Dönüş: {'ok': bool, 'output': str (yakalanan stdout), 'result': Any | None,
    'error': str | None, 'elapsed_s': float}. HER çağrı `AI_LAB_LOG_FILE`'a loglanır."""
    stdout_buf = io.StringIO()
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            g = _sandbox_globals(inputs)
            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(code_str, "<nihai_ai_lab>", "exec"), g)  # noqa: S102 — kısıtlı beyaz-liste namespace
            outcome["result"] = g.get("result")
        except BaseException as e:  # noqa: BLE001 — sandbox hatası çağırana METİN olarak dönmeli
            outcome["exc"] = e

    thread = threading.Thread(target=_target, daemon=True)
    t0 = time.monotonic()
    thread.start()
    thread.join(max(0.01, float(timeout_s)))
    elapsed = time.monotonic() - t0

    if thread.is_alive():
        _best_effort_interrupt(thread.ident)
        row: dict[str, Any] = {"ok": False, "output": stdout_buf.getvalue(), "result": None,
                               "error": f"zaman aşımı (>{timeout_s}s)", "elapsed_s": elapsed}
    elif "exc" in outcome:
        exc = outcome["exc"]
        row = {"ok": False, "output": stdout_buf.getvalue(), "result": None,
              "error": f"{type(exc).__name__}: {exc}", "elapsed_s": elapsed}
    else:
        row = {"ok": True, "output": stdout_buf.getvalue(), "result": outcome.get("result"),
              "error": None, "elapsed_s": elapsed}

    _log_run("sandbox", {
        "code": str(code_str)[:_CODE_LOG_TRUNC], "ok": row["ok"],
        "output_summary": str(row["output"])[:_OUTPUT_LOG_TRUNC], "error": row["error"],
        "elapsed_s": elapsed,
    })
    return row


# ---------------------------------------------------------------------------
# Hazır deney şablonları — GÜVENİLİR kod (sandbox'a GİRMEZ), tamamen deterministik
# ---------------------------------------------------------------------------
def _run_vol_model_karsilastirma(closes: Any, opens: Any = None, highs: Any = None,
                                 lows: Any = None, window: int = 20) -> dict[str, Any]:
    """4 volatilite tahmincisi: close-to-close, EWMA, kayan-pencere gerçekleşen-vol, GARCH(1,1)
    öngörüsü (`quant_math` sarımı — bkz. plan WP-C tablosu). OHLC verilirse (opsiyonel)
    Parkinson/Garman-Klass/Yang-Zhang de `extra_ohlc_estimates`'e eklenir (bonus, 4'e dahil değil)."""
    arr = np.asarray(closes, dtype=float)
    if arr.size < 30:
        return {"error": "yetersiz veri (>=30 kapanış gerekir)", "not": _NOT_KENAR_IDDIASI_DEGIL}
    logret = quant_math.log_returns(arr)
    estimates: dict[str, float | None] = {
        "close_to_close": quant_math.vol_close_to_close(arr),
        "ewma": quant_math.ewma_vol(logret),
    }
    win = min(int(window), logret.size)
    estimates["rolling_realized"] = float(np.std(logret[-win:], ddof=1)) if win > 1 else None
    garch = quant_math.garch11_fit(logret)
    estimates["garch11"] = quant_math.garch11_forecast(garch, horizon=1) if garch is not None else None

    extra: dict[str, float | None] = {}
    if opens is not None and highs is not None and lows is not None:
        extra["parkinson"] = quant_math.vol_parkinson(highs, lows)
        extra["garman_klass"] = quant_math.vol_garman_klass(opens, highs, lows, arr)
        extra["yang_zhang"] = quant_math.vol_yang_zhang(opens, highs, lows, arr)

    valid = [v for v in estimates.values() if v is not None]
    spread = float(max(valid) - min(valid)) if len(valid) >= 2 else None
    return {
        "n": int(arr.size), "window": win, "estimates": estimates,
        "extra_ohlc_estimates": extra or None, "spread_max_min": spread,
        "not": _NOT_KENAR_IDDIASI_DEGIL,
    }


def _to_returns_matrix(panel: Any) -> tuple[list[str], np.ndarray] | None:
    """DataFrame (kolon=sembol) VEYA dict[str, ArrayLike] → (semboller, log-getiri matrisi).
    Uyumsuz/yetersiz veri → None (çökme yok)."""
    if isinstance(panel, pd.DataFrame):
        symbols = [str(c) for c in panel.columns]
        try:
            prices = panel.to_numpy(dtype=float)
        except (TypeError, ValueError):
            return None
    elif isinstance(panel, dict):
        symbols = [str(k) for k in panel]
        try:
            prices = np.column_stack([np.asarray(v, dtype=float) for v in panel.values()])
        except ValueError:
            return None
    else:
        return None
    if prices.ndim != 2 or prices.shape[0] < 3 or prices.shape[1] < 2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(np.log(prices), axis=0)
    if not np.all(np.isfinite(rets)):
        return None
    return symbols, rets


def _run_korelasyon_rejimi(panel: Any, window: int = 60) -> dict[str, Any]:
    """Kayan-pencere ortalama ikili-korelasyon özeti + Ledoit-Wolf kovaryans kondisyon
    sayısı (`quant_math.ledoit_wolf_cov`). Rejim etiketi (dusuk/orta/yuksek_korelasyon)
    SABİT-EŞİKLİ, tanımlayıcıdır — bir sinyal/karar İMASI TAŞIMAZ."""
    parsed = _to_returns_matrix(panel)
    if parsed is None:
        return {"error": "panel geçersiz (>=2 sembol, >=3 satır, DataFrame veya dict gerekir)",
               "not": _NOT_KENAR_IDDIASI_DEGIL}
    symbols, rets = parsed
    n = rets.shape[0]
    win = max(10, min(int(window), n))
    windows: list[dict[str, Any]] = []
    for start in range(0, n - win + 1, win):
        chunk = rets[start:start + win]
        corr = np.corrcoef(chunk, rowvar=False)
        iu = np.triu_indices_from(corr, k=1)
        mean_corr = float(np.mean(corr[iu])) if iu[0].size else float("nan")
        windows.append({"start": start, "end": start + win, "mean_pairwise_corr": mean_corr})

    valid_means = [w["mean_pairwise_corr"] for w in windows if math.isfinite(w["mean_pairwise_corr"])]
    overall_mean = float(np.mean(valid_means)) if valid_means else float("nan")
    if not math.isfinite(overall_mean):
        regime = "belirsiz"
    elif overall_mean < 0.2:
        regime = "dusuk_korelasyon"
    elif overall_mean < 0.5:
        regime = "orta_korelasyon"
    else:
        regime = "yuksek_korelasyon"

    cov = quant_math.ledoit_wolf_cov(rets)
    cond = float(np.linalg.cond(cov)) if cov is not None else None
    return {
        "symbols": symbols, "n_obs": n, "window": win, "n_windows": len(windows),
        "windows": windows, "overall_mean_corr": overall_mean, "regime": regime,
        "ledoit_wolf_condition_number": cond,
        "not": _NOT_KENAR_IDDIASI_DEGIL,
    }


def _run_olay_etudu(closes: Any, event_indices: Any, pre: int = 10, post: int = 10) -> dict[str, Any]:
    """CAR/AAR (kümülatif/ortalama anormal getiri) şablonu — ORTALAMA-DÜZELTMELİ basit
    model (piyasa-modeli KULLANILMAZ; anormal = ham getiri − olay-öncesi pencere ortalaması).
    `event_indices`: log-getiri dizisindeki (uzunluk = len(closes)-1) olay bar indeksleri."""
    arr = np.asarray(closes, dtype=float)
    if arr.size < 3:
        return {"error": "yetersiz veri (>=3 kapanış gerekir)", "not": _NOT_KENAR_IDDIASI_DEGIL}
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.diff(np.log(arr))
    n = logret.size
    try:
        events = sorted({int(i) for i in event_indices if 0 <= int(i) < n})
    except (TypeError, ValueError):
        return {"error": "event_indices tam-sayıya çevrilemedi", "not": _NOT_KENAR_IDDIASI_DEGIL}
    if not events:
        return {"error": "geçerli olay indeksi yok (event_indices boş/kapsam dışı)",
               "not": _NOT_KENAR_IDDIASI_DEGIL}

    profiles: list[np.ndarray] = []
    used_events: list[int] = []
    for ev in events:
        lo, hi = ev - pre, ev + post
        if lo - pre < 0 or hi >= n:
            continue
        window_ret = logret[lo:hi + 1]
        baseline = float(np.mean(logret[lo - pre:lo]))
        if not math.isfinite(baseline):
            continue
        profiles.append(window_ret - baseline)
        used_events.append(ev)
    if not profiles:
        return {"error": "hiçbir olay için tam pencere yok (pre/post sınırları veri dışına taşıyor)",
               "not": _NOT_KENAR_IDDIASI_DEGIL}

    profiles_arr = np.vstack(profiles)
    aar = np.mean(profiles_arr, axis=0)
    car_per_event = np.sum(profiles_arr, axis=1)
    return {
        "n_events": len(used_events), "event_indices_used": used_events,
        "pre": int(pre), "post": int(post),
        "aar": aar.tolist(), "car_per_event": car_per_event.tolist(),
        "car_mean": float(np.mean(car_per_event)),
        "car_std": float(np.std(car_per_event, ddof=1)) if car_per_event.size > 1 else 0.0,
        "model": "ortalama-düzeltmeli (piyasa-modeli KULLANILMAZ — basit tanımlayıcı)",
        "not": _NOT_KENAR_IDDIASI_DEGIL,
    }


def _run_dagilim_testi(returns: Any) -> dict[str, Any]:
    """Normallik (Jarque-Bera) + çarpıklık/basıklık + kuyruk-oranı raporu (scipy.stats)."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return {"error": "yetersiz veri (>=8 gözlem gerekir)", "not": _NOT_KENAR_IDDIASI_DEGIL}
    skewness = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr, fisher=True))
    jb_stat, jb_p = stats.jarque_bera(arr)
    p05, p95 = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
    tail_ratio = float(p95 / abs(p05)) if p05 != 0 else None
    return {
        "n": int(arr.size), "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "skewness": skewness, "excess_kurtosis": kurt,
        "jarque_bera_stat": float(jb_stat), "jarque_bera_p": float(jb_p),
        "normal_mi_p05": bool(jb_p > 0.05),
        "tail_ratio_p95_over_abs_p5": tail_ratio,
        "not": _NOT_KENAR_IDDIASI_DEGIL,
    }


def _run_ozellik_tarama(X: Any, y: Any, feature_names: Any = None, seed: int = 42) -> dict[str, Any]:
    """Özellik-başına karşılıklı-bilgi (`quant_math.mutual_info`) + permütasyon-önemi
    (sklearn `RandomForestClassifier` + `permutation_importance`, sabit `seed` → deterministik)
    sıralaması. `feature_names` verilmezse `f0..f{n-1}`."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.shape[0] or X_arr.shape[0] < 10:
        return {"error": "X/y boyutları uyumsuz veya yetersiz örnek (>=10 satır gerekir)",
               "not": _NOT_KENAR_IDDIASI_DEGIL}
    n_feat = X_arr.shape[1]
    if feature_names is not None and len(list(feature_names)) == n_feat:
        names = [str(nm) for nm in feature_names]
    else:
        names = [f"f{i}" for i in range(n_feat)]

    mi_scores: dict[str, float] = {}
    for j, nm in enumerate(names):
        mi = quant_math.mutual_info(X_arr[:, j], y_arr)
        mi_scores[nm] = float(mi) if mi is not None else float("nan")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    perm_scores: dict[str, float]
    try:
        clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=seed)
        clf.fit(X_arr, y_arr)
        perm = permutation_importance(clf, X_arr, y_arr, n_repeats=10, random_state=seed)
        perm_scores = dict(zip(names, (float(v) for v in perm.importances_mean), strict=True))
    except ValueError as e:
        logger.debug(f"[ai.lab] permutation_importance başarısız: {e}")
        perm_scores = dict.fromkeys(names, float("nan"))

    ranking = sorted(names, key=lambda nm: perm_scores[nm] if math.isfinite(perm_scores[nm]) else -math.inf,
                     reverse=True)
    return {
        "features": names, "mutual_info": mi_scores, "permutation_importance": perm_scores,
        "ranking_by_permutation_importance": ranking,
        "not": _NOT_KENAR_IDDIASI_DEGIL,
    }


_TEMPLATES: dict[str, Any] = {
    "vol_model_karsilastirma": _run_vol_model_karsilastirma,
    "korelasyon_rejimi": _run_korelasyon_rejimi,
    "olay_etudu": _run_olay_etudu,
    "dagilim_testi": _run_dagilim_testi,
    "ozellik_tarama": _run_ozellik_tarama,
}


def run_experiment(name: str, **kwargs: Any) -> dict[str, Any]:
    """Hazır DETERMİNİSTİK deney şablonlarından birini çalıştırır (modül docstring'i
    listesi — hiçbiri sandbox'a girmez, GÜVENİLİR kendi kodumuzdur). Bilinmeyen `name`
    → ValueError (programcı hatası — API sözleşmesi ihlali). Şablon içi hata (yanlış
    şekil/tip) yakalanıp `{'error': ..., 'not': ...}` olarak döner (çökme yok). Her çağrı
    `AI_LAB_LOG_FILE`'a loglanır (`kind='experiment'`)."""
    fn = _TEMPLATES.get(name)
    if fn is None:
        raise ValueError(f"bilinmeyen deney: {name!r} (geçerli: {sorted(_TEMPLATES)})")
    t0 = time.monotonic()
    try:
        result = fn(**kwargs)
    except (TypeError, ValueError, KeyError) as e:
        result = {"error": f"{type(e).__name__}: {e}", "not": _NOT_KENAR_IDDIASI_DEGIL}
    elapsed = time.monotonic() - t0
    ok = "error" not in result
    _log_run("experiment", {"name": name, "ok": ok, "error": result.get("error"), "elapsed_s": elapsed})
    return result


# ---------------------------------------------------------------------------
# LLM-destekli analiz — FAIL-SOFT (router.py sözleşmesi)
# ---------------------------------------------------------------------------
def _template_answer(question: str, context: dict[str, Any]) -> str:
    """LLM yokken dönen DETERMİNİSTİK metin (rastgelelik/zaman-damgası YOK — aynı girdi
    → aynı çıktı, `ask_llm_analysis` fail-soft testinin dayanağı)."""
    keys = ", ".join(sorted(str(k) for k in context)) if context else "yok"
    return (f"[ŞABLON YANIT — LLM kullanılamıyor] Soru kaydedildi: {question!r}. "
           f"Bağlamda bulunan anahtarlar: {keys}. Bu şablon yanıt YALNIZ tanımlayıcıdır; "
           "kenar/karar iması taşımaz — gerçek yorum için ANTHROPIC_API_KEY/GEMINI_API_KEY "
           "gerekir (nihai.ai.router fail-soft zinciri).")


def ask_llm_analysis(question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """`question`+`context`'i `router.complete("judgment", ...)` ile sorar; FAIL-SOFT
    (anahtar yoksa/çağrı None dönerse — router.py sözleşmesi — DETERMİNİSTİK şablon
    yanıta düşülür). Dönüş HER ZAMAN {'question','answer','source' ('llm'|'template'),
    'not'} — `not` alanı HİÇBİR yolda kenar iddiası TAŞIMAZ (dürüstlük guard'ı)."""
    ctx = dict(context or {})
    prompt = (f"Soru: {question}\nBağlam (JSON, kısaltılmış): "
             f"{json.dumps(ctx, ensure_ascii=False, default=str)[:2000]}\n"
             "Yalnız tanımlayıcı/açıklayıcı bir analiz ver; 'kenar bulundu' gibi ifadeler kullanma.")
    answer = router.complete("judgment", prompt, max_tokens=600)
    if answer is None:
        return {"question": question, "answer": _template_answer(question, ctx),
               "source": "template", "not": _NOT_KENAR_IDDIASI_DEGIL}
    return {"question": question, "answer": answer, "source": "llm", "not": _NOT_KENAR_IDDIASI_DEGIL}
