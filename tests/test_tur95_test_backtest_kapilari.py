"""Tur-95 kilidi — "backtest + benchmark + yapılması gereken tüm testler" turunun bulguları.

Tur projenin tüm doğrulama katmanını koşturdu (5.090 test, ruff, `mypy .`, MainHub, 2.640
görevlik backtest tekrar-üretimi, derin portföy backtest'i, walk-forward, maliyet stresi, AI
karar katmanı λ süpürmesi, canlı hat replay'i, benchmark). Üç kusur çıktı; üçü de bir ÖLÇÜMÜN
ya da BELGENİN söylediği ile yaptığı arasındaki farktır:

  * **BULGU-95-1** — belgelenmiş iki komut çalışmıyordu. `tools/sync_mainhub.py` docstring'i
    olmayan bir kontrol bayrağını (argparse rc=2) ve olmayan bir test dosyasını; pano
    (`api_quant`) operatöre `dolt_options` modülünün hiç sahip olmadığı bir CLI bayrağını
    öğütlüyordu. Tur-61 kapısı yalnız MODÜLÜN varlığını soruyordu, BAYRAĞI değil.
  * **BULGU-95-2** — Monte Carlo `ruin_probability` nihai sermayeden sayılıyordu. Karıştırma
    sırayı değiştirir, toplamı değil → her yol aynı noktada biter, değer ya 0 ya 1 çıkar.
    Madde 52 kapısı (`gate.MC_RUIN_LOW`) ve tarayıcı iflas filtresi TAM bu alanı okuyordu:
    nihai sermaye pozitifken kapı yol içinde iflas eden sıralamaları hiç göremezdi.
  * **BULGU-95-3** — kill-switch KALICIDIR; sıfır-kenarlı bir seride çoğu kez tam 30. işlemde
    tetiklenir ve dönemin kalanını girişsiz bırakır. Varsayılan portföy koşuları 35-50 işlemden
    sonra 2 yılın ~%97'sinde işlem yapmıyordu ve sonuç bunu HİÇBİR alanda söylemiyordu.

Ek kilit: backtest motorunun SAYISAL davranışı (sentetik veride ALTIN değer). Önceki testler
motoru "koşuyor mu" düzeyinde sınıyordu; bir strateji/risk değişikliği sayıları SESSİZCE
değiştirebilirdi. BİLİNÇLİ bir değişiklik altın değeri güncellemeyi gerektirir — yeni değer ve
gerekçesi o turun raporuna yazılır (bu turda 2.640 görevlik koşu dünkü tabanla BİT-DENK çıktı).
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nihai.execution.backtest as bt

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ BULGU-95-1: CLI belgesi
_PAKET = ("nihai", "research", "tools")
# `python -m X --a` · `& $venvPy -m X --a` · `-Argument "-m X --a"` (sarmalayıcılar `python`
# sözcüğü OLMADAN çağırır — ilk tarama yalnız `python -m` arıyordu ve 23 ps1 çağrısını görmüyordu)
_KULLANIM = re.compile(r"(?:^|[\s\"'(&])-m\s+((?:nihai|research|tools)(?:\.\w+)*)([^\n`|;)\"]*)")
# `@("-m", "X", "--a")` · `-ArgumentList "-m","X","--a"`
_DIZI = re.compile(r"[\"']-m[\"']\s*,\s*[\"']((?:nihai|research|tools)(?:\.\w+)*)[\"']"
                   r"((?:\s*,\s*[\"'][^\"']*[\"'])*)")
_BAYRAK = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")


def _modul_dosyalari(mod: str) -> list[Path]:
    """`python -m mod` hangi dosyayı koşturur: modül dosyası ya da paketin __main__/__init__'i."""
    yol = ROOT / Path(mod.replace(".", "/"))
    if yol.with_suffix(".py").exists():
        return [yol.with_suffix(".py")]
    return [p for p in (yol / "__main__.py", yol / "__init__.py") if p.exists()]


def _argparse_bayraklari(dosyalar: list[Path]) -> set[str] | None:
    """ast ile `add_argument` seçenek dizgileri (import YOK). argparse hiç yoksa None."""
    kume: set[str] = set()
    var = False
    for p in dosyalar:
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument":
                var = True
                kume |= {a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    return kume if var else None


def _kullanimlar(metin: str):
    """(satır, modül, [bayraklar]) — bir satırda birden çok komut varsa kuyruk ilkinde kesilir."""
    for ln, satir in enumerate(metin.splitlines(), 1):
        for m in list(_KULLANIM.finditer(satir)) + list(_DIZI.finditer(satir)):
            kuyruk = re.split(r"\s-m\s|&&|\|\|", m.group(2).split("#")[0])[0]
            kuyruk = re.sub(r"\[([^\]]*)\]", r" \1 ", kuyruk)          # [--apply] → --apply
            yield ln, m.group(1).rstrip("."), _BAYRAK.findall(kuyruk)


def _cli_sorunlari(metinler: dict[str, str]) -> list[str]:
    sorun: list[str] = []
    for ad, metin in metinler.items():
        for ln, mod, bayraklar in _kullanimlar(metin):
            dosyalar = _modul_dosyalari(mod)
            if not dosyalar:
                sorun.append(f"{ad}:{ln} MODUL_YOK {mod}")
                continue
            kume = _argparse_bayraklari(dosyalar)
            for f in bayraklar:
                b = f.split("=")[0]
                if b in ("-h", "--help"):
                    continue
                if kume is None:
                    sorun.append(f"{ad}:{ln} ARGPARSE_YOK {mod} {b}")
                elif not (b in kume or (b.startswith("--") and sum(k.startswith(b) for k in kume) == 1)):
                    # argparse benzersiz öneki kabul eder (allow_abbrev) → tekil önek geçerlidir
                    sorun.append(f"{ad}:{ln} BAYRAK_YOK {mod} {b}")
    return sorun


def _kaynaklar() -> dict[str, str]:
    """Operatörün ve zamanlanmış görevlerin OKUDUĞU yüzeyler: kod (docstring/yorum/ipucu dizgisi),
    CLAUDE.md, README.md, kökteki .ps1'ler. `docs/`, CHANGELOG ve OPTIMIZASYON_LOG TARİHSEL kayıttır
    (Tur-61 raporu eski kusuru bilerek alıntılar) → kapsam dışı."""
    out: dict[str, str] = {}
    for pkg in _PAKET:
        for p in (ROOT / pkg).rglob("*.py"):
            if "__pycache__" not in p.parts:
                out[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8", errors="replace")
    for p in [ROOT / "CLAUDE.md", ROOT / "README.md", *sorted(ROOT.glob("*.ps1"))]:
        out[p.name] = p.read_text(encoding="utf-8", errors="replace")
    return out


def test_belgelenen_cli_komutlarinin_BAYRAKLARI_modulde_var():
    """Belgedeki her `-m <modül> --bayrak` o modülün argparse'ında olmalı (BULGU-95-1)."""
    sorun = _cli_sorunlari(_kaynaklar())
    assert not sorun, "belgelenen komut çalışmaz:\n" + "\n".join(sorun)


def test_cli_kapisi_GERCEKTEN_eliyor():
    """Kapı ölü olmasın (Tur-60): bu turun iki kusuru, iki çağrı biçimi ve olmayan modül
    yakalanmalı; üretimdeki doğru komut geçmeli."""
    sahte = {
        "a.py": "python -m tools.sync_mainhub --check     # sürüklenme var mı?",
        "b.py": "(`python -m nihai.data.dolt_options --bulk`)",
        "c.ps1": '$x = Start-Svc "k" @("-m", "research.forward_harness", "--boyle-bayrak-yok") 1',
        "d.md": "`python -m nihai.main`",
        "e.ps1": "$out = & $venvPy -m research.forward_harness --resolve --snapshot --report 2>&1 |",
        "f.md": "python -m tools.sync_mainhub [--apply]",
    }
    sorun = _cli_sorunlari(sahte)
    assert any(s.startswith("a.py") and "BAYRAK_YOK" in s and "--check" in s for s in sorun)
    assert any(s.startswith("b.py") and "ARGPARSE_YOK" in s for s in sorun)
    assert any(s.startswith("c.ps1") and "--boyle-bayrak-yok" in s for s in sorun)
    assert any(s.startswith("d.md") and "MODUL_YOK" in s for s in sorun)
    assert not any(s.startswith(("e.ps1", "f.md")) for s in sorun), sorun


def test_ps1_uretim_cagrilari_GERCEKTEN_taraniyor():
    """Gece görevinin komutu ve dizi biçimli servis başlatması tarayıcının görüş alanında."""
    fh = (ROOT / "forward-harness.ps1").read_text(encoding="utf-8", errors="replace")
    assert ("research.forward_harness", ["--resolve", "--snapshot", "--report"]) in \
        [(m, b) for _, m, b in _kullanimlar(fh)]
    pa = (ROOT / "proje-ac.ps1").read_text(encoding="utf-8", errors="replace")
    assert ("nihai.execution.main", ["--serve-web"]) in [(m, b) for _, m, b in _kullanimlar(pa)]


def test_BULGU_95_1_iki_komut_duzeltildi():
    sync = (ROOT / "tools" / "sync_mainhub.py").read_text(encoding="utf-8")
    assert "sync_mainhub --check" not in sync                 # olmayan bayrak
    assert "test_tur58_mainhub_parity" not in sync            # olmayan test dosyası
    assert "test_tur58_hub_koprusu.py" in sync and (ROOT / "tests" / "test_tur58_hub_koprusu.py").exists()
    quant = (ROOT / "nihai" / "web" / "api_quant.py").read_text(encoding="utf-8")
    assert "dolt_options --bulk" not in quant
    assert "D.bulk_csv('volatility_history')" in quant
    # ipucunun indirdiği dosya, uç zincirinin okuduğu dosyadır (aynı ad kalıbı)
    from nihai.data import dolt_options, etf_holdings
    assert 'f"{table}__{safe_ref}.csv"' in inspect.getsource(dolt_options.bulk_csv)
    assert 'f"volatility_history__' in inspect.getsource(etf_holdings.implied_correlation_frame)


# ------------------------------------------------------------ BULGU-95-2: MC iflası
def test_BULGU_95_2_iflas_YOL_ICINDE_sayilir():
    """[-120, +200], sermaye 100: +200 önce gelirse yol 100→300→180 (iflas yok); −120 önce
    gelirse 100→−20→180 (yolda iflas). Nihai sermaye iki sırada da 180 → eski tanım 0,0 derdi."""
    mc = bt.monte_carlo_paths([-120.0, 200.0], 100.0, 400, seed=3)
    assert mc["final_equity_p05"] == mc["final_equity_p50"] == mc["final_equity_p95"] == 180.0
    assert 0.3 < mc["ruin_probability"] < 0.7


def test_yol_iflasi_NIHAI_iflastan_asla_kucuk_degil():
    """Nihai-iflas yolu yolda da iflastır → yeni tanım eskisinden yalnız SIKI olabilir."""
    rng = np.random.default_rng(5)
    for _ in range(30):
        pnls = [float(x) for x in rng.normal(-5, 60, int(rng.integers(3, 40)))]
        eq = float(rng.uniform(50, 400))
        mc = bt.monte_carlo_paths(pnls, eq, 200, seed=int(rng.integers(0, 1000)))
        assert mc["ruin_probability"] >= (1.0 if mc["final_equity_p50"] <= 0 else 0.0)
        # karıştırma toplamı değiştirmez; yalnız toplama SIRASI son bitlerde oynar (Tur-67: ~1e-13)
        assert mc["final_equity_p05"] == pytest.approx(mc["final_equity_p95"], abs=1e-6)


def test_madde52_kapisi_yol_icinde_iflasi_ARTIK_goruyor():
    """Nihai sermaye pozitifken MC_RUIN_LOW eskiden HİÇBİR yolda düşemezdi."""
    from nihai.execution import gate
    mc = bt.monte_carlo_paths([-120.0, 200.0], 100.0, 400, seed=3)
    rep = gate.evaluate_backtest_dict({"trades_count": 999, "win_rate": 1.0, "expectancy": 1.0,
                                       "max_drawdown_pct": 0.0, "monte_carlo": mc})
    kontrol = {c.name: c.passed for c in rep.checks}
    assert kontrol.pop("MC_RUIN_LOW") is False
    assert all(kontrol.values())                                  # yalnız iflas kontrolü düşer


def test_eski_nihai_iflas_sayimi_geri_gelmedi():
    assert "for v in final_equities if v <= 0" not in inspect.getsource(bt.monte_carlo_paths)


def test_cli_raporu_nihai_sermayeyi_TEK_deger_basar(capsys):
    r = bt.backtest_symbol("SYN", "1d", df=_sentetik(1, 900, 0.0, 0.02), strategy_mode="event_breakout",
                           initial_equity=100_000.0, mc_iters=50)
    bt.print_report(r)
    out = capsys.readouterr().out
    assert "Final NLV (siradan bagimsiz)" in out and "Final NLV  p05/p50/p95" not in out
    assert "Iflas (yolda <= 0)" in out


# ------------------------------------------------------------ BULGU-95-3: kill-switch
def _sentetik(seed: int, n: int, drift: float, vol: float) -> pd.DataFrame:
    """Tohumlu GBM günlük barı (global np.random'a dokunmaz)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = np.maximum(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), 1.0)
    op = close * (1 + rng.normal(0, vol / 3, n))
    high = np.maximum(op, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(op, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close, "volume": v}, index=idx)


def _tek(seed: int, **kw) -> bt.BacktestResult:
    return bt.backtest_symbol("SYN", "1d", df=_sentetik(seed, 900, 0.0, 0.02), strategy_mode="event_breakout",
                              initial_equity=100_000.0, **kw)


def _portfoy(**kw) -> tuple[bt.BacktestResult, dict]:
    data = {s: _sentetik(i, 700, 0.0003, 0.02) for i, s in enumerate(("AAA", "BBB", "CCC"), 40)}
    return bt.backtest_portfolio(list(data), "1d", data=data, initial_equity=100_000.0, **kw), data


def test_BULGU_95_3_kill_switch_ani_ve_girissiz_payi_tasinir():
    df = _sentetik(2, 900, 0.0, 0.02)
    r = _tek(2)
    assert r.kill_switch_at is not None and 0 < (r.kill_switch_idle_pct or 0) < 100
    i = df.index.get_loc(pd.Timestamp(r.kill_switch_at))
    assert r.kill_switch_idle_pct == round(100 * (len(df) - 1 - i) / (len(df) - 1 - 60), 2)
    assert "KILL_SWITCH@" in r.short_summary()


def test_kill_switch_YALNIZ_sonraki_girisleri_keser():
    """Tetiklenene kadarki işlemler kill-switch kapalı koşuyla BİREBİR aynı; fark yalnız sonrası."""
    acik, kapali = _tek(2), _tek(2, disable_kill_switch=True)
    assert kapali.kill_switch_at is None and kapali.kill_switch_idle_pct is None
    assert kapali.trades_count > acik.trades_count
    assert kapali.trade_pnls[: acik.trades_count] == acik.trade_pnls


def test_kill_switch_tetiklenmezse_alanlar_None():
    r = _tek(1)
    assert r.trades_count >= 30 and r.kill_switch_at is None and r.kill_switch_idle_pct is None
    assert "KILL_SWITCH@" not in r.short_summary()


def test_portfoy_kill_switch_ani_tasinir():
    r, data = _portfoy(strategy_mode="event_breakout")
    all_ts = sorted(set().union(*[set(d.index) for d in data.values()]))
    k = all_ts.index(pd.Timestamp(r.kill_switch_at))
    assert r.kill_switch_idle_pct == round(100 * (len(all_ts) - k) / len(all_ts), 2)
    kapali, _ = _portfoy(strategy_mode="event_breakout", disable_kill_switch=True)
    assert kapali.kill_switch_at is None and kapali.trades_count > r.trades_count


def _kapanis_kaydi(monkeypatch) -> list:
    """Her kapanışı (çıkış zamanı, PnL) sırasıyla kaydeden PositionManager (backtest'e yamalı)."""
    kayit: list = []
    asil = bt.PositionManager

    class _Kayitli(asil):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.add_on_close(lambda ce: kayit.append((pd.Timestamp(ce.exit_time), ce.realized_pnl)))

    monkeypatch.setattr(bt, "PositionManager", _Kayitli)
    return kayit


def _tetik_zamani(kayit: list) -> pd.Timestamp | None:
    """Kill-switch kuralını backtest'ten BAĞIMSIZ yeniden kur: son ≤50 işlem, ≥30, ortalama < 0."""
    son: deque = deque(maxlen=50)
    for t, pnl in kayit:
        son.append(pnl)
        if len(son) >= 30 and sum(son) / len(son) < 0:
            return t
    return None


def test_kill_switch_ani_KURALDAN_bagimsiz_dogrulanir(monkeypatch):
    """Raporlanan an, kuralın kill-switch'siz koşunun kapanış sırasından yeniden kurulan tetiğinin
    bir bar öncesidir (tetikleyen kapanış bar i+1'de işlenir, bar i kararının girişi kesilir).
    Öz-referanslı kontrol (an ↔ boşta payı) kayık bir indeksi YAKALAYAMAZDI (mutasyon M5 kaçtı)."""
    kayit = _kapanis_kaydi(monkeypatch)
    _tek(2, disable_kill_switch=True)
    tetik = _tetik_zamani(kayit)
    df = _sentetik(2, 900, 0.0, 0.02)
    acik = _tek(2)
    assert tetik is not None
    assert df.index.get_loc(tetik) - 1 == df.index.get_loc(pd.Timestamp(acik.kill_switch_at))

    kayit.clear()
    _, data = _portfoy(strategy_mode="event_breakout", disable_kill_switch=True)
    tetik = _tetik_zamani(kayit)
    r, _ = _portfoy(strategy_mode="event_breakout")
    all_ts = sorted(set().union(*[set(d.index) for d in data.values()]))
    assert tetik is not None
    assert all_ts.index(tetik) - 1 == all_ts.index(pd.Timestamp(r.kill_switch_at))


# ------------------------------------------------------------ ALTIN değer + belirlenimcilik
def test_backtest_motoru_ALTIN_deger_tek_sembol():
    """Değişirse: strateji/risk/icra davranışı değişmiştir. Bilinçliyse yeni değeri gerekçesiyle
    tur raporuna yaz ve burayı güncelle; bilinçsizse bir gerileme yakaladın."""
    r = _tek(1)
    assert (r.trades_count, round(r.final_equity or 0.0, 2)) == (46, 128931.70)


def test_backtest_motoru_ALTIN_deger_portfoy():
    r, _ = _portfoy(strategy_mode="ensemble")
    assert (r.trades_count, round(r.final_equity or 0.0, 2)) == (85, 127522.87)


def test_backtest_BELIRLENIMCI():
    """Aynı girdi → alan alan aynı sonuç (2.640 görevlik günler-arası koşunun süreç-içi hâli)."""
    a, b = dataclasses.asdict(_tek(3)), dataclasses.asdict(_tek(3))
    assert a == b
