"""
shadow_decision_backtest.py — GÖLGE KARAR MOTORUNUN ileri-getiri ölçümü (2026-07-26).

Neden: 2026-07-26 denetimi B-1'de karar motorunun YAPISAL olarak %100 ABSTAIN ürettiği
ölçüldü; düzeltmeden (yan-koşullu prior) sonra motor ilk kez yön üretebiliyor. Bu harness
"peki ürettiği yönler DOĞRU mu?" sorusunu ölçer.

Yöntem: watchlist'teki her sembol için günlük barlar çekilir; her karar barında YALNIZ
o bara kadarki kapanışlarla `decision.decide(mode zorla off → hiçbir dosyaya yazılmaz)`
çağrılır, sonra `horizon` bar ileri getiri ölçülür.

DÜRÜSTLÜK KAPILARI (ACI DERS #1/#3/#7):
  1. `--exec-lag N`  : sinyalden N bar SONRA girilir. §22 dersi — kenarın ilk barda
                       yoğunlaşması reversal/bid-ask artefaktıdır, kointegrasyon değil.
  2. TABAN karşılaştırması: motorun isabeti, aynı barlarda "hep LONG" ve "hep SHORT"
                       triviyal kurallarıyla YAN YANA raporlanır. Taban oranından
                       FAZLALIK yoksa sonuç kenar DEĞİLDİR (ACI DERS #8 mantığı).
  3. PRIOR KİRLİLİĞİ : `decision` prior'ı `data/ai_learning_state.json`'dan okur ve o
                       durum bu backtest penceresinin İÇİNDEKİ canlı sonuçlardan
                       türetilmiştir → bu ölçüm İN-SAMPLE'dır. Rapor bunu her koşuda
                       açıkça yazar; OOS iddiası ASLA yapılmaz.
  4. Maliyet         : `--cost-bps` çift-yönlü işlem maliyeti getiriden düşülür.

Kullanım (repo kökünden):
    python -m research.shadow_decision_backtest --days 180 --horizon 5
    python -m research.shadow_decision_backtest --days 180 --horizon 5 --exec-lag 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nihai.ai import decision as decision_mod  # noqa: E402
from nihai.ai import shadow_runner  # noqa: E402
from nihai.console import ensure_utf8_stdout

_MIN_WARMUP = 60          # karar için asgari kapanış sayısı (rejim/CVaR hesapları)


def _decide_off(symbol: str, strategy: str, closes: np.ndarray,
                side_hint: str | float | None, lam: float | None = None) -> Any:
    """`decide()` YAZMA-SIZ çağrı: AI_DECISION_MODE geçici 'off' (brain._demo deseni).

    `lam` verilirse `AI_UTILITY_LAMBDA` da geçici olarak o değere kurulur (λ süpürmesi;
    decision.py env'i ÇAĞRI ANINDA okur, bu yüzden yeniden içe aktarma gerekmez)."""
    prev = os.environ.get("AI_DECISION_MODE")
    prev_lam = os.environ.get("AI_UTILITY_LAMBDA")
    os.environ["AI_DECISION_MODE"] = "off"
    if lam is not None:
        os.environ["AI_UTILITY_LAMBDA"] = str(lam)
    try:
        return decision_mod.decide(symbol=symbol, strategy=strategy, features={},
                                   closes=closes, side_hint=side_hint)
    finally:
        for k, v in (("AI_DECISION_MODE", prev), ("AI_UTILITY_LAMBDA", prev_lam)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run(days: int = 180, horizon: int = 5, exec_lag: int = 0, cost_bps: float = 5.0,
        max_symbols: int = 30, step: int = 1) -> dict:
    syms = shadow_runner._resolve_watchlist()[:max_symbols]
    if not syms:
        return {"hata": "watchlist boş"}
    try:
        from nihai.data.data_provider import get_provider
        provider = get_provider()
    except Exception as e:  # noqa: BLE001
        return {"hata": f"sağlayıcı alınamadı: {e}"}

    lookback = max(days + horizon + exec_lag + _MIN_WARMUP + 10, 260)
    rows: list[dict] = []
    n_sym_ok = 0
    for sym in syms:
        df = shadow_runner._fetch_bars(provider, sym, lookback_bars=lookback)
        if df is None:
            continue
        closes = shadow_runner._closes_from(df, sym)
        if closes is None or closes.size < _MIN_WARMUP + horizon + exec_lag + 5:
            continue
        n_sym_ok += 1
        n = closes.size
        start = max(_MIN_WARMUP, n - days - horizon - exec_lag)
        for i in range(start, n - horizon - exec_lag, max(1, step)):
            hist = closes[: i + 1]                       # YALNIZ karar barına kadar
            # side_hint: karar barındaki kısa-vadeli momentum işareti (aday tarafı).
            mom = float(hist[-1] / hist[-6] - 1.0) if hist.size >= 6 else 0.0
            d = _decide_off(sym, "event_breakout", hist, mom)
            entry = float(closes[i + exec_lag])
            exit_ = float(closes[i + exec_lag + horizon])
            if not (np.isfinite(entry) and np.isfinite(exit_)) or entry <= 0:
                continue
            fwd = exit_ / entry - 1.0
            rows.append({"symbol": sym, "i": i, "action": d.action,
                         "conviction": float(d.conviction), "posterior": float(d.posterior),
                         "prior_src": _prior_src(d), "fwd": fwd, "mom": mom})
    if not rows:
        return {"hata": "hiç karar üretilemedi", "n_sembol": n_sym_ok}

    df_r = pd.DataFrame(rows)
    cost = 2.0 * cost_bps / 1e4
    out: dict[str, Any] = {
        "n_sembol": n_sym_ok, "n_karar": len(df_r), "horizon_bars": horizon,
        "exec_lag": exec_lag, "cost_bps": cost_bps, "gun": days,
        "aksiyon_dagilimi": df_r["action"].value_counts().to_dict(),
        "prior_kaynagi": df_r["prior_src"].value_counts().to_dict(),
    }

    sided = df_r[df_r["action"].isin(["LONG_BIAS", "SHORT_BIAS"])].copy()
    out["n_yonlu"] = len(sided)
    out["abstain_orani"] = round(float((df_r["action"] == "ABSTAIN").mean()), 4)
    if len(sided):
        sign = np.where(sided["action"] == "LONG_BIAS", 1.0, -1.0)
        signed = sided["fwd"].to_numpy() * sign
        net = signed - cost
        out["motor"] = _stats(signed, net)
        for act in ("LONG_BIAS", "SHORT_BIAS"):
            sub = sided[sided["action"] == act]
            if len(sub):
                s = sub["fwd"].to_numpy() * (1.0 if act == "LONG_BIAS" else -1.0)
                out[act.lower()] = _stats(s, s - cost)

    # TABAN kuralları — AYNI barlarda triviyal karşılaştırma (ACI DERS #8)
    all_fwd = df_r["fwd"].to_numpy()
    out["taban_hep_long"] = _stats(all_fwd, all_fwd - cost)
    out["taban_hep_short"] = _stats(-all_fwd, -all_fwd - cost)
    out["taban_momentum"] = _stats(all_fwd * np.sign(df_r["mom"].to_numpy()),
                                   all_fwd * np.sign(df_r["mom"].to_numpy()) - cost)
    if "motor" in out:
        out["FAZLALIK_vs_hep_long"] = round(
            out["motor"]["ort_getiri_pct"] - out["taban_hep_long"]["ort_getiri_pct"], 4)
        out["FAZLALIK_vs_hep_short"] = round(
            out["motor"]["ort_getiri_pct"] - out["taban_hep_short"]["ort_getiri_pct"], 4)
    out["UYARI"] = ("Prior `data/ai_learning_state.json`'dan okunur ve bu pencerenin canlı "
                    "sonuçlarından türemiştir → bu ölçüm İN-SAMPLE'dır, OOS kenar kanıtı DEĞİLDİR "
                    "(ACI DERS #3/#4).")
    return out


DEFAULT_LAMBDAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]


def _collect(days: int, horizon: int, exec_lag: int, max_symbols: int,
             step: int) -> list[dict]:
    """Karar bağlamlarını BİR KEZ topla (bar çekimi λ'dan bağımsızdır).

    Her bağlam: karar barına kadarki kapanışlar + o bardan itibaren gerçekleşen ileri
    getiri. λ süpürmesi bunları yeniden kullanır → 8 λ için 8 kez bar çekilmez."""
    syms = shadow_runner._resolve_watchlist()[:max_symbols]
    from nihai.data.data_provider import get_provider
    provider = get_provider()
    lookback = max(days + horizon + exec_lag + _MIN_WARMUP + 10, 260)
    ctxs: list[dict] = []
    for sym in syms:
        df = shadow_runner._fetch_bars(provider, sym, lookback_bars=lookback)
        if df is None:
            continue
        closes = shadow_runner._closes_from(df, sym)
        if closes is None or closes.size < _MIN_WARMUP + horizon + exec_lag + 5:
            continue
        n = closes.size
        start = max(_MIN_WARMUP, n - days - horizon - exec_lag)
        for i in range(start, n - horizon - exec_lag, max(1, step)):
            hist = closes[: i + 1]
            entry = float(closes[i + exec_lag])
            exit_ = float(closes[i + exec_lag + horizon])
            if not (np.isfinite(entry) and np.isfinite(exit_)) or entry <= 0:
                continue
            mom = float(hist[-1] / hist[-6] - 1.0) if hist.size >= 6 else 0.0
            ctxs.append({"symbol": sym, "i": i, "hist": hist, "mom": mom,
                         "fwd": exit_ / entry - 1.0})
    return ctxs


def _eval_lambda(ctxs: list[dict], lam: float, cost: float) -> dict:
    """Tek λ için motor kararlarını üret ve TABAN kurallarıyla yan yana ölç."""
    acts, fwds, moms = [], [], []
    for c in ctxs:
        d = _decide_off(c["symbol"], "event_breakout", c["hist"], c["mom"], lam=lam)
        acts.append(d.action)
        fwds.append(c["fwd"])
        moms.append(c["mom"])
    a = np.asarray(acts)
    f = np.asarray(fwds, dtype=float)
    m = np.asarray(moms, dtype=float)
    sided = np.isin(a, ["LONG_BIAS", "SHORT_BIAS"])
    out: dict[str, Any] = {
        "lambda": lam, "n": int(f.size),
        "abstain_orani": round(float((a == "ABSTAIN").mean()), 4) if f.size else None,
        "n_yonlu": int(sided.sum()),
        "taban_hep_long_pct": round(float(f.mean() - cost) * 100, 4) if f.size else None,
        "taban_momentum_pct": (round(float((f * np.sign(m)).mean() - cost) * 100, 4)
                               if f.size else None),
    }
    if sided.any():
        sign = np.where(a[sided] == "LONG_BIAS", 1.0, -1.0)
        signed = f[sided] * sign
        out["motor"] = _stats(signed, signed - cost)
        out["FAZLALIK_vs_hep_long"] = round(
            out["motor"]["ort_net_getiri_pct"] - (out["taban_hep_long_pct"] or 0.0), 4)
        out["FAZLALIK_vs_momentum"] = round(
            out["motor"]["ort_net_getiri_pct"] - (out["taban_momentum_pct"] or 0.0), 4)
    return out


def lambda_sweep(days: int = 180, horizon: int = 5, exec_lag: int = 1,
                 cost_bps: float = 5.0, max_symbols: int = 30, step: int = 1,
                 lambdas: list[float] | None = None, split: float = 0.6) -> dict:
    """O1 — λ'yı TAHMİNLE değil ÖLÇÜMLE seç. Zaman-bölmeli: λ ilk `split` payında seçilir,
    kalan payda DOĞRULANIR. Tek bölmede pozitif fazlalık kabul edilmez (ACI DERS #3).

    Red kriteri gerçekleşirse (hiçbir λ OOS'ta pozitif fazlalık vermiyor) sonuç
    "λ=0.5'te kal" olur ve bu ölçülmüş bir gerekçedir, kaza değil."""
    lambdas = lambdas or DEFAULT_LAMBDAS
    ctxs = _collect(days, horizon, exec_lag, max_symbols, step)
    if not ctxs:
        return {"hata": "karar bağlamı toplanamadı"}
    ctxs.sort(key=lambda c: (c["i"], str(c["symbol"])))
    k = int(len(ctxs) * split)
    ic, oos = ctxs[:k], ctxs[k:]
    cost = 2.0 * cost_bps / 1e4

    tablo: list[dict[str, Any]] = []
    for lam in lambdas:
        tablo.append({"lambda": lam,
                      "SECIM": _eval_lambda(ic, lam, cost),
                      "DOGRULAMA": _eval_lambda(oos, lam, cost)})

    def _f(row: dict[str, Any], bolme: str, alan: str, vars: float = -9e9) -> float:
        return float((row.get(bolme) or {}).get(alan, vars) or vars)

    # λ*: SEÇİM bölmesinde fazlalığı en yüksek olan (yön üretebilenler arasında)
    aday = [r for r in tablo if _f(r, "SECIM", "n_yonlu", 0.0) > 0]
    best: dict[str, Any] | None = (
        max(aday, key=lambda r: _f(r, "SECIM", "FAZLALIK_vs_hep_long")) if aday else None)
    dogrulandi = bool(best and _f(best, "DOGRULAMA", "FAZLALIK_vs_hep_long") > 0
                      and _f(best, "DOGRULAMA", "n_yonlu", 0.0) > 0)
    # Tur-86 (BULGU-86-2): "hiçbir λ pozitif fazlalık vermedi" cümlesi, DOĞRULAMA bölmesinde
    # HİÇBİR λ yön üretmediğinde de doğrudur — ama boş kümeden kurulmuş bir hükümdür. O
    # durumda fazlalık ÖLÇÜLEMEZ ve "ABSTAIN ölçülmüş doğru davranış" denemez: yön kapısı
    # (prior CI tabanı > 0,5, kapı-a) λ'dan bağımsızdır ve λ'yı değiştirmek hiçbir şeyi
    # değiştirmez. Karar (λ=0.5'te kal) aynıdır; değişen yalnız hükmün GEREKÇESİdir.
    konusan_dog = [r for r in tablo if _f(r, "DOGRULAMA", "n_yonlu", 0.0) > 0]
    if dogrulandi:
        kod = "DOGRULANDI"
        verdikt = (f"λ*={best.get('lambda') if best else None} DOĞRULAMA bölmesinde de pozitif "
                   "fazlalık verdi — .env'e yazılabilir (yine de canlı gölge kanıtı gerekir)")
    elif not konusan_dog:
        kod = "KONUSAN_LAMBDA_YOK"
        verdikt = ("Hiçbir λ (λ=0 dahil) DOĞRULAMA bölmesinde yön üretmedi → fazlalık "
                   "ÖLÇÜLEMEDİ. %100 ABSTAIN'i λ eşiği değil prior güven kapısı (yön için CI "
                   "tabanı > 0,5) belirliyor; bu koşu 'ABSTAIN doğru' hükmü VERMEZ.")
    else:
        kod = "KAL"
        verdikt = ("Hiçbir λ doğrulama bölmesinde pozitif fazlalık vermedi → λ=0.5'te KAL. "
                   "Üretimdeki %100 ABSTAIN artık ölçülmüş doğru davranıştır, kaza değil.")
    return {
        "parametre": {"gun": days, "horizon": horizon, "exec_lag": exec_lag,
                      "cost_bps": cost_bps, "n_baglam": len(ctxs), "split": split,
                      "n_secim": len(ic), "n_dogrulama": len(oos)},
        "tablo": tablo,
        "lambda_yildiz": (best.get("lambda") if best else None),
        "VERDIKT_KOD": kod,
        "VERDIKT": verdikt,
        "UYARI": ("Prior `data/ai_learning_state.json`'dan gelir ve pencerenin canlı "
                  "sonuçlarından türemiştir → SEÇİM bölmesi in-sample'dır; DOĞRULAMA bölmesi "
                  "de prior kirliliğinden tam arınmış değildir (ACI DERS #3/#4)."),
    }


def _prior_src(d: Any) -> str:
    for ev in getattr(d, "evidence", []) or []:
        if str(ev).startswith("Prior (kaynak="):
            return str(ev).split("kaynak=", 1)[1].split(")", 1)[0]
    return "?"


def _stats(signed: np.ndarray, net: np.ndarray) -> dict:
    signed = np.asarray(signed, dtype=float)
    net = np.asarray(net, dtype=float)
    signed = signed[np.isfinite(signed)]
    net = net[np.isfinite(net)]
    if signed.size == 0:
        return {"n": 0}
    wins = net[net > 0]
    losses = net[net < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else None
    sd = float(signed.std(ddof=1)) if signed.size > 1 else 0.0
    return {
        "n": int(signed.size),
        "isabet": round(float((signed > 0).mean()), 4),
        "ort_getiri_pct": round(float(signed.mean()) * 100, 4),
        "ort_net_getiri_pct": round(float(net.mean()) * 100, 4),
        "profit_factor_net": round(pf, 4) if pf is not None else None,
        "t_stat": round(float(signed.mean() / (sd / np.sqrt(signed.size))), 3) if sd > 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gölge karar motoru ileri-getiri ölçümü")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--horizon", type=int, default=5)
    # Tur-86 (BULGU-86-1): varsayılan None = çağrılan fonksiyonun KENDİ varsayılanı
    # (`run` 0, `lambda_sweep` 1 — dürüstlük kapısı #1). Eskiden CLI 0'ı HER ZAMAN geçiriyor,
    # `--lambda-sweep` koşusunu exec-lag kapısı kapalı ölçüyordu.
    ap.add_argument("--exec-lag", type=int, default=None)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--max-symbols", type=int, default=30)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--lambda-sweep", action="store_true",
                    help="O1: λ'yı zaman-bölmeli ölçümle seç")
    ap.add_argument("--lambdas", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    ap.add_argument("--split", type=float, default=0.6)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    lag: dict[str, int] = {} if a.exec_lag is None else {"exec_lag": a.exec_lag}
    if a.lambda_sweep:
        res: dict = lambda_sweep(days=a.days, horizon=a.horizon,
                                 cost_bps=a.cost_bps, max_symbols=a.max_symbols, step=a.step,
                                 lambdas=[float(x) for x in a.lambdas.split(",") if x.strip()],
                                 split=a.split, **lag)
    else:
        res = run(days=a.days, horizon=a.horizon,
                  cost_bps=a.cost_bps, max_symbols=a.max_symbols, step=a.step, **lag)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(txt)
    _write_kpi(res, a.lambda_sweep)
    return 0


def _write_kpi(res: dict[str, Any], is_sweep: bool) -> None:
    """O3 — `selfcheck.exam_production`'ın okuduğu artefaktı yaz (data/uretim_kpi.json).

    Süpürmede ÜRETİM λ'sının (0.5) satırı yazılır; "λ gevşetilirse ne olurdu" değil,
    "bugün üretimde ne oluyor" ölçülmelidir."""
    from datetime import UTC, datetime

    from nihai.config import DATA_DIR
    src: dict[str, Any] = res
    en_iyi: dict[str, Any] = {}
    if is_sweep:
        tablo: list[dict[str, Any]] = list(res.get("tablo") or [])
        rows = [r for r in tablo if abs(float(r.get("lambda", -1)) - 0.5) < 1e-9]
        src = (rows[0].get("DOGRULAMA") or {}) if rows else {}
        # Üretim λ'sı hiç yön üretmiyorsa fazlalık ölçülemez; "λ gevşetilseydi ne olurdu"
        # bilgisi ayrı alanda taşınır ki karne 'ölçülmedi' ile 'ölçüldü, kötü' ayırabilsin.
        konusan = [r for r in tablo if (r.get("DOGRULAMA") or {}).get("n_yonlu", 0) > 0]
        if konusan:
            b: dict[str, Any] = max(
                konusan, key=lambda r: (r.get("DOGRULAMA") or {}).get("FAZLALIK_vs_hep_long", -9e9))
            dog: dict[str, Any] = b.get("DOGRULAMA") or {}
            en_iyi = {"lambda": b.get("lambda"),
                      "FAZLALIK_vs_hep_long": dog.get("FAZLALIK_vs_hep_long"),
                      "n_yonlu": dog.get("n_yonlu")}
    kpi = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "FAZLALIK_vs_hep_long": src.get("FAZLALIK_vs_hep_long"),
        "FAZLALIK_vs_momentum": src.get("FAZLALIK_vs_momentum"),
        "n_yonlu": src.get("n_yonlu"), "abstain_orani": src.get("abstain_orani"),
        "kaynak": "lambda_sweep(λ=0.5, DOĞRULAMA bölmesi)" if is_sweep else "run()",
        "en_iyi_konusan_lambda": en_iyi or None,
        "lambda_yildiz": res.get("lambda_yildiz"), "verdikt": res.get("VERDIKT"),
        # Tur-86: ölçüm KOŞULU ve makine-okunur hüküm, okunacağı yere (selfcheck) taşınır —
        # 2026-08-10 artefaktı hangi exec_lag/gün ile ölçüldüğünü söyleyemiyordu (Tur-63).
        "verdikt_kod": res.get("VERDIKT_KOD"), "parametre": res.get("parametre"),
    }
    try:
        (DATA_DIR / "uretim_kpi.json").write_text(
            json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[backtest] üretim KPI artefaktı yazılamadı: {e}", file=sys.stderr)


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
