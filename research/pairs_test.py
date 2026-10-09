"""pairs_test.py — Pairs / kointegrasyon POINT-IN-TIME (survivorship-temiz) deneyi + dürüst ölçüm.

YENİ kenar-ailesi adayı. Bugüne dek ölçülen üç aile (PEAD, meta-labeling, cross-sectional
momentum) PIT'te KENAR YOK çıktı (CLAUDE.md ACI DERSLER §1-6). Pairs, fiyat-gösterge
zamanlamasından da kesit-momentumdan da FARKLI bir kaynaktır: iki kointegre hissenin
spread'inin ortalamaya dönmesi (istatistiksel arbitraj). Bu script onu AYNI disiplinle ölçer.

RESEARCH-ONLY. Üretim koduna dokunmaz, "kenar bulundu" İLAN ETMEZ. Verdikt yalnız
"kanıt kapısını geçti/geçmedi"; geçse bile sonuç "kenar" değil "forward-shadow adayı"dır
(canlı gölge izleme sonraki adım — META_GATE deseni).

Deney (ön-kayıtlı çerçeve):
- Walk-forward: formation=252 işlem günü → takip eden trade=63 gün; 63'er gün kaydır
  (örtüşmeyen trade pencereleri) → OOS yapısı (purged-CV'nin bu tasarımdaki karşılığı).
- Her formation başında: PIT S&P500 üyeler ∩ alpaca_hist'te formation penceresi TAM dolu
  semboller. Aday çiftler = log-getiri |korelasyon| en yüksek ~500 → Engle-Granger coint
  (log fiyatlar) → p<eşik → en iyi 20 çift/pencere (p'ye göre).
- Hedge oranı: (A) formation-OLS statik beta; (B) Kalman dinamik beta (durum=[beta,alpha]
  random-walk, saf numpy). İki varyant AYRI raporlanır.
- Spread z: statikte formation mean/std (trade'de YENİDEN ölçekleme YOK = look-ahead yok);
  Kalman'da innovation/√S. Giriş |z|>=2.0, çıkış |z|<=0.5, stop |z|>=3.5, time-stop pencere
  sonu. Dolar-nötr 1:|beta| bacaklar; sinyal t kapanışında, uygulama t+1 kapanış-to-kapanış
  (same-bar fill YOK). Maliyet bacak-yön başına 5 bps (round-trip ~20 bps).
- KANIT KAPISI: (1) günlük getiriden Sharpe + deflated.dsr (n_trials=denenen konfig sayısı);
  (2) pbo.cscv_pbo (T×N = günlük getiri × konfig); (3) removed-since (delist) alt-kümesi AYRI
  tablo (PEAD dersi §13.5); (4) yıl kırılımı.

    python -m research.pairs_test --grid          # 12-konfig ızgara + DSR + PBO (ana koşu)
    python -m research.pairs_test --variant a --z-entry 2.0 --cost-bps 5   # tek konfig
    python -m research.pairs_test --grid --limit 200   # hızlı deneme (evreni kısıtla)
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import numpy as np
import pandas as pd

# statsmodels FAIL-SOFT (Tur-26). Bu modül-düzeyi import KATIYDI ve `tests/test_pairs_math.py`
# onu import ettiği için, statsmodels kurulu olmayan bir ortamda pytest TOPLAMA fazında
# durup **tüm 2178 testi** düşürüyordu (ölçüldü: "Interrupted: 1 error during collection").
# Yani opsiyonel bir RESEARCH bağımlılığının yokluğu tüm kalite kapısını kör ediyordu.
# requirements.txt zaten "runtime'da kullanılmaz" diyor; proje konvansiyonu da opsiyonel
# bağımlılıkta importorskip/guard (bkz. xgboost, quantstats, empyrical, torch, adfuller).
# `coint` modül-düzeyi AD olarak korunur — testler `monkeypatch.setattr("research.
# pairs_test.coint", ...)` ile onu değiştiriyor; yer tutucu yalnız GERÇEKTEN çağrılırsa
# ve statsmodels yoksa, yönergeli bir hata verir.
try:
    from statsmodels.tsa.stattools import coint
except ImportError:  # pragma: no cover - opsiyonel research bağımlılığı
    def coint(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "statsmodels kurulu değil — pairs/kointegrasyon research'ü için gerekli: "
            "pip install 'statsmodels>=0.14'"
        )

from research.deflated import dsr, sharpe_moments
from research.pbo import cscv_pbo, pbo_report

# Veri dizini: alpaca_history.HIST_DIR ile AYNI çözüm (o modülü import etmeden — alpaca SDK
# yükünden kaçınmak için). data/'dan yalnız OKUMA (üretim verisine yazma yok).
try:
    from nihai.config import DATA_DIR
    _HIST_DIR = Path(os.getenv("ALPACA_HIST_DIR", str(DATA_DIR / "alpaca_hist")))
except Exception:   # pragma: no cover - config yoksa göreli yol
    _HIST_DIR = Path(os.getenv("ALPACA_HIST_DIR", "data/alpaca_hist"))

_TRADING_DAYS = 252   # yıllıklandırma / Sharpe için


# ---------------------------------------------------------------------------
# Veri: cache'lenmiş günlük CSV'leri kapanış-fiyatı paneline oku (ağsız, PIT-uyumlu)
# ---------------------------------------------------------------------------
def load_panel(symbols: list[str] | None = None) -> pd.DataFrame:
    """alpaca_hist/{SYM}_1d.csv dosyalarını ortak tarih indeksinde kapanış paneline topla.

    Delist edilmiş (removed-since) semboller de dosyada olduğundan panel survivorship-temiz;
    her sembol yalnız işlem gördüğü tarihlerde dolu (delist sonrası NaN → çift hizalamada düşer).
    """
    paths = sorted(glob.glob(str(_HIST_DIR / "*_1d.csv")))
    want = set(symbols) if symbols is not None else None
    series: dict[str, pd.Series] = {}
    for p in paths:
        sym = os.path.basename(p)[:-7]   # "..._1d.csv" son 7 karakteri at
        if want is not None and sym not in want:
            continue
        try:
            df = pd.read_csv(p, usecols=["timestamp", "close"],
                             parse_dates=["timestamp"], index_col="timestamp")
            if not df.empty:
                series[sym] = df["close"]
        except Exception:
            continue
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


# ---------------------------------------------------------------------------
# PIT üyelik: pit_universe.sp500_as_of rewind mantığının birebir aynası, TEK Wikipedia
# taramasıyla (15+ pencere = 15+ tarama yerine 1). _current_and_changes bir kez cache'lenir.
# ---------------------------------------------------------------------------
_PIT_CACHE: tuple[list[str], pd.DataFrame] | None = None


def _pit_tables() -> tuple[list[str], pd.DataFrame]:
    global _PIT_CACHE
    if _PIT_CACHE is None:
        from nihai.data.pit_universe import _current_and_changes
        _PIT_CACHE = _current_and_changes()
    return _PIT_CACHE


def members_as_of(as_of: pd.Timestamp) -> set[str]:
    """`as_of` tarihindeki S&P500 üyeliği (çıkarılanlar dahil). pit_universe.sp500_as_of aynası."""
    from nihai.data.pit_universe import _norm
    cur, changes = _pit_tables()
    members = set(cur)
    future = changes[changes["date"] > as_of]
    sort_cols = ["date"] + [c for c in ("added", "removed") if c in future.columns]
    future = future.sort_values(by=sort_cols, ascending=False, kind="mergesort")
    for _, row in future.iterrows():
        added = _norm(row["added"]) if "added" in row and pd.notna(row.get("added")) else ""
        removed = _norm(row["removed"]) if "removed" in row and pd.notna(row.get("removed")) else ""
        if added and added in members:
            members.discard(added)
        if removed:
            members.add(removed)
    return {m for m in members if m and m != "NAN" and len(m) <= 6}


def current_members() -> set[str]:
    """BUGÜNKÜ üyeler (removed-since etiketi için); pit_universe._norm ile normalize."""
    from nihai.data.pit_universe import _norm
    cur, _ = _pit_tables()
    return {_norm(s) for s in cur}


# ---------------------------------------------------------------------------
# Hedge matematiği: OLS beta/alpha, statik z, Kalman dinamik beta z
# ---------------------------------------------------------------------------
def ols_beta_alpha(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]:
    """log_a ~ beta·log_b + alpha en-küçük-kareler. Dönüş (beta, alpha)."""
    x = np.column_stack([log_b, np.ones_like(log_b)])
    coef, *_ = np.linalg.lstsq(x, log_a, rcond=None)
    return float(coef[0]), float(coef[1])


def static_spread_stats(log_a: np.ndarray, log_b: np.ndarray,
                        beta: float, alpha: float) -> tuple[float, float]:
    """Formation spread'inin (resid = log_a − beta·log_b − alpha) ortalama + std'si (ddof=0)."""
    resid = log_a - beta * log_b - alpha
    return float(resid.mean()), float(resid.std())


def static_z(log_a: np.ndarray, log_b: np.ndarray, beta: float, alpha: float,
             mu: float, sigma: float) -> np.ndarray:
    """Statik z: spread'i FORMATION mu/sigma ile ölçekle. Trade penceresinde YENİDEN
    ölçekleme YOK → look-ahead yok (mu,sigma yalnız formation'dan gelir)."""
    if sigma <= 0.0:
        return np.zeros_like(log_a)
    return (log_a - beta * log_b - alpha - mu) / sigma


def kalman_beta_z(log_a: np.ndarray, log_b: np.ndarray, *, beta0: float, alpha0: float,
                  obs_var: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Kalman filtresi (saf numpy). Durum x=[beta,alpha] random-walk; gözlem
    log_a_t = beta_t·log_b_t + alpha_t + gürültü.

    Predict: P ← P + Q (Q = delta·I, süreç gürültüsü). Innovation e_t = log_a_t − H_t·x,
    H_t=[log_b_t, 1]; S_t = H_t·P·H_tᵀ + R (R = obs_var, gözlem gürültüsü). z_t = e_t/√S_t.
    x_0=[beta0,alpha0] formation-OLS'ten. e_t/S_t contemporaneous (t'de log_b_t de log_a_t de
    bilinir) → look-ahead yok; sinyal t→t+1 pozisyonunu belirler. Dönüş (z_dizisi, beta_dizisi)."""
    n = len(log_a)
    x = np.array([beta0, alpha0], dtype=float)
    cov = np.eye(2) * 1e-3
    q = delta * np.eye(2)
    r = max(obs_var, 1e-10)
    z = np.zeros(n)
    betas = np.zeros(n)
    for t in range(n):
        cov = cov + q
        h = np.array([log_b[t], 1.0])
        e = log_a[t] - float(h @ x)
        s = float(h @ cov @ h) + r
        z[t] = e / math.sqrt(s) if s > 0 else 0.0
        k = (cov @ h) / s
        x = x + k * e
        cov = cov - np.outer(k, h) @ cov
        betas[t] = x[0]
    return z, betas


# ---------------------------------------------------------------------------
# Tek çift simülasyonu: giriş/çıkış/stop state-machine, lag-1 uygulama, maliyet
# ---------------------------------------------------------------------------
def simulate_pair(dates: pd.DatetimeIndex, close_a: np.ndarray, close_b: np.ndarray,
                  z: np.ndarray, beta_hold: np.ndarray, *, z_entry: float, z_exit: float,
                  z_stop: float, cost_bps: float, involves_removed: bool) -> tuple[pd.Series, list[dict]]:
    """Trade penceresi (hizalanmış çift dizileri) üzerinde işlem simülasyonu.

    z[t] karar barı t'de hesaplanır (kapanış t), pozisyon t→t+1 barında tutulur (r[t+1] kazanılır)
    = same-bar fill YOK. Giriş |z|>=z_entry (z>0 → spread zengin → SHORT spread; z<0 → LONG).
    Çıkış |z|<=z_exit veya stop |z|>=z_stop; time-stop pencere sonu. Dolar-nötr 1:|beta| bacak:
    long-spread günlük getiri = (r_a − k·r_b)/(1+k), k=|giriş beta'sı| (işlem boyu sabit).
    Maliyet: açılış 2 bacak + kapanış 2 bacak, her biri cost_bps → round-trip 4·cost_bps.

    Dönüş: (günlük NET getiri Serisi [yalnız tutulan günler], tamamlanan işlem listesi)."""
    n = len(dates)
    ret_a = np.zeros(n)
    ret_b = np.zeros(n)
    ret_a[1:] = close_a[1:] / close_a[:-1] - 1.0
    ret_b[1:] = close_b[1:] / close_b[:-1] - 1.0
    cost = cost_bps * 1e-4

    daily: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []
    pos = 0          # 0 / +1 (long spread) / -1 (short spread)
    entry_idx = -1
    k = 0.0
    cum = 0.0        # aktif işlemin kümülatif net getirisi

    for t in range(n):
        zt = z[t]
        finite = math.isfinite(zt)
        if pos == 0:
            # Karar t → pozisyonu t→t+1'de tut. Son barda giriş yok (kazanç barı kalmaz).
            if finite and t < n - 1 and abs(zt) >= z_entry:
                pos = -1 if zt > 0 else 1
                entry_idx = t
                k = abs(beta_hold[t])
                cum = 0.0
        else:
            # Tutulan pozisyon t-1→t getirisini kazan (karar t-1'de verildi).
            leg = (ret_a[t] - k * ret_b[t]) / (1.0 + k) if (1.0 + k) > 0 else 0.0
            day_ret = pos * leg
            if t == entry_idx + 1:
                day_ret -= 2.0 * cost          # açılış maliyeti ilk kazanç günü
            close_now = finite and (abs(zt) <= z_exit or abs(zt) >= z_stop)
            time_stop = t == n - 1
            if close_now or time_stop:
                day_ret -= 2.0 * cost          # kapanış maliyeti
            daily[dates[t]] = daily.get(dates[t], 0.0) + day_ret
            cum += day_ret
            if close_now or time_stop:
                trades.append({
                    "entry_date": dates[entry_idx], "exit_date": dates[t],
                    "side": pos, "beta": k, "hold_days": t - entry_idx,
                    "net_ret": cum, "involves_removed": involves_removed,
                    "exit_reason": "time" if (time_stop and not close_now) else
                                   ("stop" if abs(zt) >= z_stop else "mean"),
                })
                pos = 0
                entry_idx = -1
    ser = pd.Series(daily, dtype=float).sort_index() if daily else pd.Series(dtype=float)
    return ser, trades


# ---------------------------------------------------------------------------
# Pencere çift-seçimi (konfigden BAĞIMSIZ → tek sefer hesaplanıp cache'lenir)
# ---------------------------------------------------------------------------
def select_window_pairs(panel: pd.DataFrame, t0: int, form: int, members: set[str],
                        max_cand: int) -> list[dict]:
    """Formation penceresi [t0-form:t0] için aday kointegre çiftler + p-değerleri.

    Eleme: (1) PIT üye ∩ formation TAM dolu semboller; (2) log-getiri |korelasyon| en yüksek
    max_cand çift; (3) her aday Engle-Granger coint (log fiyat). Tüm p'ler döner (p-eşiği
    aşağı-akışta uygulanır → tek coint hesabı iki eşik için yeniden kullanılır). p'ye göre sıralı.
    """
    form_df = panel.iloc[t0 - form:t0]
    ok = [c for c in form_df.columns
          if c in members and form_df[c].notna().all() and (form_df[c] > 0).all()]
    if len(ok) < 2:
        return []
    sub = form_df[ok]
    log_px = np.log(sub.values)
    rets = np.diff(log_px, axis=0)
    corr = np.corrcoef(rets.T)
    # Üst üçgen |korelasyon| → en yüksek max_cand çift.
    iu, ju = np.triu_indices(len(ok), k=1)
    order = np.argsort(-np.abs(corr[iu, ju]))[:max_cand]
    out: list[dict] = []
    for idx in order:
        i, j = int(iu[idx]), int(ju[idx])
        a, b = ok[i], ok[j]
        la, lb = log_px[:, i], log_px[:, j]
        try:
            _, pval, _ = coint(la, lb)
        except Exception:
            continue
        beta, alpha = ols_beta_alpha(la, lb)
        mu, sigma = static_spread_stats(la, lb, beta, alpha)
        out.append({"a": a, "b": b, "pval": float(pval), "beta": beta,
                    "alpha": alpha, "mu": mu, "sigma": sigma, "corr": float(corr[i, j])})
    out.sort(key=lambda d: d["pval"])
    return out


# ---------------------------------------------------------------------------
# Konfig koşusu: pencereler boyunca çift simülasyonu → günlük getiri + işlemler
# ---------------------------------------------------------------------------
def _apply_exec_lag(z: np.ndarray, beta_hold: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Sinyali `lag` bar GECİKTİR (uygulama gecikmesi robustluk kontrolü). lag=0 ön-kayıtlı
    konvansiyon (sinyal t → t+1 kapanış). lag>0: gerçekçi ek gecikme; 1-bar kısa-vade reversal /
    bid-ask sıçraması artefaktı bu testte çöker (dayanıklı kointegrasyon dönüşü çökmez)."""
    if lag <= 0 or len(z) <= lag:
        return z, beta_hold
    z2 = np.concatenate([np.zeros(lag), z[:-lag]])
    b2 = np.concatenate([np.full(lag, beta_hold[0]), beta_hold[:-lag]])
    return z2, b2


def run_config(panel: pd.DataFrame, windows: list[tuple[int, pd.Timestamp]],
               pair_cache: dict[int, list[dict]], current: set[str], *, form: int, trade: int,
               variant: str, z_entry: float, z_exit: float, z_stop: float, p_thr: float,
               cost_bps: float, max_pairs: int, kalman_delta: float,
               kalman_obs_scale: float, exec_lag: int = 0) -> dict:
    """Bir konfig için tüm pencereleri koş. variant 'a'=statik OLS, 'b'=Kalman.

    exec_lag: sinyal uygulama gecikmesi (0=ön-kayıtlı). Dönüş: {daily, trades, cov_pairs}."""
    all_daily: list[pd.Series] = []      # her çift-örneği bir Seri
    trades: list[dict] = []
    cov_pairs = 0

    for t0, _asof in windows:
        cands = [c for c in pair_cache[t0] if c["pval"] < p_thr][:max_pairs]
        cov_pairs += len(cands)
        for c in cands:
            a, b = c["a"], c["b"]
            seg = panel[[a, b]].iloc[t0:t0 + trade].dropna()
            if len(seg) < 3:
                continue
            dates = seg.index
            ca, cb = seg[a].to_numpy(), seg[b].to_numpy()
            la, lb = np.log(ca), np.log(cb)
            if variant == "a":
                z = static_z(la, lb, c["beta"], c["alpha"], c["mu"], c["sigma"])
                beta_hold = np.full(len(seg), c["beta"])
            else:
                # Kalman: formation+trade warmup; obs_var = formation resid varyansı × ölçek.
                warm = panel[[a, b]].iloc[t0 - form:t0 + trade].dropna()
                wla, wlb = np.log(warm[a].to_numpy()), np.log(warm[b].to_numpy())
                obs_var = max(c["sigma"] ** 2, 1e-10) * kalman_obs_scale
                z_full, beta_full = kalman_beta_z(wla, wlb, beta0=c["beta"], alpha0=c["alpha"],
                                                  obs_var=obs_var, delta=kalman_delta)
                mask = warm.index.isin(dates)
                z = z_full[mask]
                beta_hold = beta_full[mask]
                if len(z) != len(seg):     # hizalama güvencesi
                    continue
            z, beta_hold = _apply_exec_lag(z, beta_hold, exec_lag)
            involves_removed = (a not in current) or (b not in current)
            ser, tr = simulate_pair(dates, ca, cb, z, beta_hold, z_entry=z_entry,
                                    z_exit=z_exit, z_stop=z_stop, cost_bps=cost_bps,
                                    involves_removed=involves_removed)
            if not ser.empty:
                all_daily.append(ser)
            trades.extend(tr)

    if all_daily:
        mat = pd.concat(all_daily, axis=1, sort=True)   # tarih-sıralı (PBO satır sırası ister)
        # Portföy günlük getiri = o gün AKTİF çiftlerin eşit-ağırlık ortalaması (yoksa 0 = nakit).
        daily = mat.mean(axis=1, skipna=True)
    else:
        daily = pd.Series(dtype=float)
    return {"daily": daily, "trades": trades, "cov_pairs": cov_pairs}


# ---------------------------------------------------------------------------
# Metrik toplama
# ---------------------------------------------------------------------------
def _pf_stats(net_rets: list[float]) -> dict:
    if not net_rets:
        return {"n": 0, "wr": 0.0, "pf": None, "expectancy": 0.0}
    wins = [r for r in net_rets if r > 0]
    losses = [r for r in net_rets if r <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    return {
        "n": len(net_rets),
        "wr": round(100 * len(wins) / len(net_rets), 1),
        "pf": round(gw / gl, 2) if gl > 0 else None,
        "expectancy": round(1e4 * (sum(net_rets) / len(net_rets)), 1),   # bps/işlem
    }


def daily_sharpe(daily: pd.Series) -> tuple[float, float, dict]:
    """(yıllık Sharpe, boyutsuz günlük Sharpe, sharpe_moments) — DSR boyutsuz SR ister."""
    if daily.empty or len(daily) < 2:
        return 0.0, 0.0, {"sr": 0.0, "skew": 0.0, "kurt": 3.0, "n": len(daily)}
    mom = sharpe_moments(daily.to_numpy())
    sr = mom["sr"]
    return sr * math.sqrt(_TRADING_DAYS), sr, mom


def aggregate(result: dict) -> dict:
    """Bir konfig sonucundan tüm metrikler: genel PF/Sharpe + removed-since + yıl kırılımı."""
    trades = result["trades"]
    daily = result["daily"]
    net = [t["net_ret"] for t in trades]
    rem = [t["net_ret"] for t in trades if t["involves_removed"]]
    surv = [t["net_ret"] for t in trades if not t["involves_removed"]]
    ann, sr, _mom = daily_sharpe(daily)
    total = float((1.0 + daily).prod() - 1.0) if not daily.empty else 0.0
    out = {
        "overall": _pf_stats(net),
        "removed_since": _pf_stats(rem),
        "survivors": _pf_stats(surv),
        "sharpe_ann": round(ann, 2),
        "sharpe_dimless": sr,
        "total_return_pct": round(100 * total, 1),
        "n_days": int(len(daily)),
        "cov_pairs": result["cov_pairs"],
    }
    by_year: dict[int, list[float]] = {}
    for t in trades:
        by_year.setdefault(int(pd.Timestamp(t["entry_date"]).year), []).append(t["net_ret"])
    out["by_year"] = {y: _pf_stats(v) for y, v in sorted(by_year.items())}
    return out


# ---------------------------------------------------------------------------
# Pencere planı + çift cache
# ---------------------------------------------------------------------------
def build_windows(panel: pd.DataFrame, form: int, trade: int) -> list[tuple[int, pd.Timestamp]]:
    """Örtüşmeyen trade pencereleri: t0=form+w·trade (trade başlangıç bar indeksi), as-of=
    formation başlangıç tarihi (panel.index[t0-form]). Son kazanç barı ≤ panel sonu."""
    n = len(panel)
    windows = []
    w = 0
    while form + w * trade + trade <= n:
        t0 = form + w * trade
        windows.append((t0, panel.index[t0 - form]))
        w += 1
    return windows


def build_pair_cache(panel: pd.DataFrame, windows: list[tuple[int, pd.Timestamp]],
                     form: int, max_cand: int) -> tuple[dict[int, list[dict]], list[int]]:
    """Her pencere için aday çiftleri (p-değerleriyle) TEK sefer hesapla. Konfigler paylaşır."""
    cache: dict[int, list[dict]] = {}
    univ_sizes: list[int] = []
    for t0, asof in windows:
        members = members_as_of(pd.Timestamp(asof))
        pairs = select_window_pairs(panel, t0, form, members, max_cand)
        cache[t0] = pairs
        form_df = panel.iloc[t0 - form:t0]
        ok = [c for c in form_df.columns
              if c in members and form_df[c].notna().all()]
        univ_sizes.append(len(ok))
    return cache, univ_sizes


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------
def _fmt_pf(s: dict) -> str:
    return f"n={s['n']:<5} WR={s['wr']:>5.1f}% PF={s['pf']} E={s['expectancy']:>+8.1f}bps"


def print_config_report(name: str, agg: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"  genel        {_fmt_pf(agg['overall'])}")
    print(f"  hayatta-kalan{_fmt_pf(agg['survivors'])}")
    print(f"  removed-since{_fmt_pf(agg['removed_since'])}")
    print(f"  Sharpe(yıllık)={agg['sharpe_ann']}  toplam={agg['total_return_pct']}%  "
          f"gün={agg['n_days']}  aday-çift={agg['cov_pairs']}")
    if agg["by_year"]:
        ys = "  ".join(f"{y}:PF{v['pf']}(n{v['n']})" for y, v in agg["by_year"].items())
        print(f"  yıl: {ys}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Pairs/kointegrasyon PIT deneyi (research-only)")
    ap.add_argument("--formation", type=int, default=252)
    ap.add_argument("--trade", type=int, default=63)
    ap.add_argument("--max-cand-pairs", type=int, default=500, help="korelasyon ön-eleme tavanı")
    ap.add_argument("--max-pairs", type=int, default=20, help="pencere başına en iyi K çift (p)")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="bacak-yön başına bps")
    ap.add_argument("--variant", default="both", choices=["a", "b", "both"])
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--z-stop", type=float, default=3.5)
    ap.add_argument("--p-thr", type=float, default=0.05)
    # delta = süreç/gözlem gürültü oranını belirler. 1e-4'te beta AŞIRI uyarlanır → innovation
    # z_std ~0.15'e çöker, ön-kayıtlı ±2 eşiği asla tetiklenmez (dejenere). 1e-6: Q<<R → beta
    # YAVAŞ değişen (tasarım niyeti), z standart ölçekte. Seçim z-ölçeğine göre (PnL'e DEĞİL).
    ap.add_argument("--kalman-delta", type=float, default=1e-6)
    ap.add_argument("--kalman-obs-scale", type=float, default=1.0)
    ap.add_argument("--exec-lag", type=int, default=0,
                    help="uygulama gecikmesi bar (0=ön-kayıtlı; 1=gerçekçi ek gecikme robustluk)")
    ap.add_argument("--limit", type=int, default=0, help="ilk N sembol (hızlı deneme; 0=hepsi)")
    ap.add_argument("--grid", action="store_true", help="12-konfig ızgara + DSR + PBO + maliyet duyarlılığı")
    ap.add_argument("--pbo-s", type=int, default=10, help="CSCV blok sayısı (çift, 4..16)")
    args = ap.parse_args()

    t_start = time.time()
    panel = load_panel()
    if args.limit > 0:
        panel = panel[list(panel.columns[:args.limit])]
    if panel.empty:
        print("HATA: panel boş (alpaca_hist/*_1d.csv bulunamadı).")
        return
    try:
        current = current_members()
    except Exception as e:
        print(f"UYARI: PIT üyelik alınamadı ({e}); survivorship-temiz test YAPILAMAZ.")
        return

    windows = build_windows(panel, args.formation, args.trade)
    print(f"=== Pairs PIT deneyi | panel {panel.shape[1]} sembol × {panel.shape[0]} bar "
          f"({panel.index[0].date()}→{panel.index[-1].date()}) ===")
    print(f"formation={args.formation} trade={args.trade} | {len(windows)} örtüşmeyen pencere | "
          f"max-cand={args.max_cand_pairs} max-pairs={args.max_pairs}")

    print("Çift-seçimi hesaplanıyor (coint; pencereler arası tek sefer)...", flush=True)
    pair_cache, univ_sizes = build_pair_cache(panel, windows, args.formation, args.max_cand_pairs)
    avg_univ = round(sum(univ_sizes) / len(univ_sizes), 1) if univ_sizes else 0
    n_removed_in_univ = _count_removed_coverage(pair_cache, current)
    print(f"Kapsama: ort. evren {avg_univ} sembol/pencere ({min(univ_sizes)}..{max(univ_sizes)}); "
          f"aday-çiftlerde {n_removed_in_univ} removed-since sembol göründü.")

    variants = ["a", "b"] if args.variant == "both" else [args.variant]

    if not args.grid:
        for v in variants:
            res = run_config(panel, windows, pair_cache, current, form=args.formation,
                             trade=args.trade, variant=v, z_entry=args.z_entry, z_exit=args.z_exit,
                             z_stop=args.z_stop, p_thr=args.p_thr, cost_bps=args.cost_bps,
                             max_pairs=args.max_pairs, kalman_delta=args.kalman_delta,
                             kalman_obs_scale=args.kalman_obs_scale, exec_lag=args.exec_lag)
            print_config_report(f"variant={v} z_entry={args.z_entry} p<{args.p_thr} "
                                f"cost={args.cost_bps}bps", aggregate(res))
        print(f"\nruntime {time.time() - t_start:.0f}s")
        return

    # ---- Izgara: 2 varyant × z-giriş{1.5,2.0,2.5} × p{0.01,0.05} = 12 konfig ----
    z_entries = [1.5, 2.0, 2.5]
    p_thrs = [0.01, 0.05]
    grid = [(v, ze, pt) for v in ["a", "b"] for ze in z_entries for pt in p_thrs]
    print(f"\n### IZGARA: {len(grid)} konfig (DSR n_trials={len(grid)}, PBO T×N matrisi) ###")

    daily_cols: dict[str, pd.Series] = {}
    grid_aggs: dict[str, dict] = {}
    for v, ze, pt in grid:
        res = run_config(panel, windows, pair_cache, current, form=args.formation,
                         trade=args.trade, variant=v, z_entry=ze, z_exit=args.z_exit,
                         z_stop=args.z_stop, p_thr=pt, cost_bps=args.cost_bps,
                         max_pairs=args.max_pairs, kalman_delta=args.kalman_delta,
                         kalman_obs_scale=args.kalman_obs_scale, exec_lag=args.exec_lag)
        key = f"{v}|z{ze}|p{pt}"
        daily_cols[key] = res["daily"]
        grid_aggs[key] = aggregate(res)
        print_config_report(key, grid_aggs[key])

    # Ortak günlük eksende T×N matris (aktif-yoksa 0 = nakit).
    axis = pd.Index(sorted(set().union(*[s.index for s in daily_cols.values() if not s.empty])))
    mat = pd.DataFrame({k: s.reindex(axis).fillna(0.0) for k, s in daily_cols.items()})
    # Boş kolon (hiç işlem) DSR/PBO'yu bozar → sabit-0 kolonlar sharpe 0 verir, sorun değil.

    # DSR: her konfigin boyutsuz günlük Sharpe'ı; var_trials = konfigler-arası Sharpe varyansı.
    sr_dimless = {k: grid_aggs[k]["sharpe_dimless"] for k in daily_cols}
    var_trials = float(np.var(list(sr_dimless.values())))
    n_days = int(len(axis))
    print("\n### KANIT KAPISI ###")
    print(f"var_trials (konfigler-arası boyutsuz-Sharpe varyansı) = {var_trials:.5g}, "
          f"n_trials={len(grid)}, T={n_days} gün")
    print(f"\n{'konfig':<14} {'Sharpe_yıl':>10} {'boyutsuz':>9} {'DSR':>7}")
    for k in daily_cols:
        mom = sharpe_moments(mat[k].to_numpy())
        d = dsr(mom["sr"], mom["n"], len(grid), var_trials, skew=mom["skew"], kurt=mom["kurt"])
        print(f"{k:<14} {grid_aggs[k]['sharpe_ann']:>10} {mom['sr']:>9.3f} {d:>7.3f}")

    # PBO: CSCV, T×N günlük getiri matrisi.
    try:
        pbo = cscv_pbo(mat.to_numpy(), s=args.pbo_s, metric="sharpe")
        print("\n" + pbo_report(pbo))
    except Exception as e:
        print(f"\nPBO hesaplanamadı: {e}")

    # ---- Ana tablo: varyant × maliyet (z_entry=2.0, p<0.05) ----
    print("\n### ANA TABLO: varyant × maliyet (z_entry=2.0, p<0.05) ###")
    print(f"{'varyant':<8} {'maliyet':>8} {'PF':>6} {'E(bps)':>8} {'Sharpe':>7} {'DSR':>7} "
          f"{'rem-PF':>7} {'surv-PF':>8}")
    for v in ["a", "b"]:
        for cb in [0.0, 5.0, 10.0]:
            res = run_config(panel, windows, pair_cache, current, form=args.formation,
                             trade=args.trade, variant=v, z_entry=2.0, z_exit=args.z_exit,
                             z_stop=args.z_stop, p_thr=0.05, cost_bps=cb, max_pairs=args.max_pairs,
                             kalman_delta=args.kalman_delta, kalman_obs_scale=args.kalman_obs_scale,
                             exec_lag=args.exec_lag)
            agg = aggregate(res)
            _, srd, mom = daily_sharpe(res["daily"])
            d = dsr(srd, mom["n"], len(grid), var_trials, skew=mom["skew"], kurt=mom["kurt"])
            o, rem, surv = agg["overall"], agg["removed_since"], agg["survivors"]
            print(f"{v:<8} {cb:>6.0f}bps {str(o['pf']):>6} {o['expectancy']:>+8.1f} "
                  f"{agg['sharpe_ann']:>7} {d:>7.3f} {str(rem['pf']):>7} {str(surv['pf']):>8}")

    # ---- ZAMANLAMA DAYANIKLILIĞI (BELİRLEYİCİ): sinyali +1/+2 bar geciktir (z2.0, p<0.05, 5bps) ----
    # 1-bar kısa-vade reversal / bid-ask sıçraması artefaktı bu testte ÇÖKER; dayanıklı
    # kointegrasyon dönüşü yalnız hafif bozulur. Kenar bu barda mı yaşıyor sorusunun cevabı.
    print("\n### ZAMANLAMA DAYANIKLILIĞI: exec_lag 0→1→2 (z_entry=2.0, p<0.05, 5bps) ###")
    print(f"{'varyant':<8} {'exec_lag':>9} {'PF':>6} {'E(bps)':>8} {'Sharpe':>7} {'n':>6}")
    for v in ["a", "b"]:
        for lag in [0, 1, 2]:
            res = run_config(panel, windows, pair_cache, current, form=args.formation,
                             trade=args.trade, variant=v, z_entry=2.0, z_exit=args.z_exit,
                             z_stop=args.z_stop, p_thr=0.05, cost_bps=5.0, max_pairs=args.max_pairs,
                             kalman_delta=args.kalman_delta, kalman_obs_scale=args.kalman_obs_scale,
                             exec_lag=lag)
            agg = aggregate(res)
            o = agg["overall"]
            print(f"{v:<8} {lag:>9} {str(o['pf']):>6} {o['expectancy']:>+8.1f} "
                  f"{agg['sharpe_ann']:>7} {o['n']:>6}")

    print(f"\nruntime {time.time() - t_start:.0f}s")


def _count_removed_coverage(pair_cache: dict[int, list[dict]], current: set[str]) -> int:
    """Aday çiftlerde görünen benzersiz removed-since sembol sayısı (kapsama kanıtı)."""
    seen: set[str] = set()
    for pairs in pair_cache.values():
        for c in pairs:
            for s in (c["a"], c["b"]):
                if s not in current:
                    seen.add(s)
    return len(seen)


if __name__ == "__main__":
    main()
