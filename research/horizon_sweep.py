"""
horizon_sweep.py — CANLI tahminlerin UFUK süpürmesi (F1, 2026-07-26 planı).

SORU: `data/predictions_log.csv`'deki 50k tahminin **tamamı** `horizon_bars=5` (1m barda
**5 dakika**) ile puanlandı ve isabet %46,8 çıktı. Bu, sinyal ailesinin gerçek kalitesi mi,
yoksa yalnızca YANLIŞ ÖLÇEKTE ölçülmüş olmasının artefaktı mı? Sistemin en büyük
ölçülmemiş serbestlik derecesi bu tek parametre.

Bu harness cevabı **geriye dönük** verir — haftalarca canlı veri beklemeye gerek yok:
`nihai/data/alpaca_history.py` ücretsiz SIP planında 1m tarihsel bar verdiği için mevcut
tahminlerin HEPSİ H ∈ {1,5,15,30,60,120,390} barda yeniden puanlanabilir.

DÜRÜSTLÜK KAPILARI (ACI DERS #1/#3/#7/#8 ve GELISTIRME_RAPORU §5):
  1. **exec-lag**    : `--exec-lags 0,1,2,5` — kenar sinyal barından SONRAKİ ilk barda
                       yoğunlaşıyorsa bu reversal/bid-ask artefaktıdır, kenar değil (§22).
  2. **Maliyet**     : `--spread-model` ile Corwin–Schultz/Roll tahmini VEYA sabit bps.
                       Medyan hareket ~13,5 bps olduğu için sıfır-maliyet varsayımı yanıltıcıdır.
  3. **Zaman bölmesi**: her (H, lag) kombinasyonu ilk yarı / ikinci yarı alt-örnekleminde
                       ayrı raporlanır. Tek bölmede pozitif olan sonuç kabul edilmez.
  4. **Kümelenmiş t** : 1m frekansta H barlık pencereler ÜST ÜSTE BİNER; ham t-istatistiği
                       bu yüzden şişer. Karar `t_kume` (sembol×gün kümelerinde) ile verilir.
  5. **Çoklu test**   : |H| × |lag| kombinasyon denenir → Šidák düzeltilmiş kritik t raporlanır
                       ve verdikt onu kullanır. 28 denemede t>2 şans eseri ~1,4 kez çıkar.
  6. **Hizalama kanıtı**: logdaki `price` ile bar kapanışı ve H=5/lag=0'da logdaki `hit`
                       yeniden üretilir (`parite`). Parite düşükse ölçüm değil HİZALAMA bozuktur.

Kabul kriteri (planla birebir): en az bir (H, exec-lag ≥ 1) kombinasyonunda net getiri
pozitif **ve** kümelenmiş t > Šidák-kritik **ve** her iki zaman yarısında da pozitif.
Red kriteri: hiçbiri → "bu sinyal ailesi bu ufuklarda ölü" (EDGE_FINDINGS §23).

Kullanım (repo kökünden):
    python -m research.horizon_sweep
    python -m research.horizon_sweep --horizons 1,5,15,30,60,120,390 --exec-lags 0,1,2 \
        --spread-model corwin --json data/horizon_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nihai.config import DATA_DIR  # noqa: E402
from nihai.console import ensure_utf8_stdout
from nihai.execution.prediction_tracker import _canon_ts  # noqa: E402

DEFAULT_HORIZONS = [1, 5, 15, 30, 60, 120, 390]
DEFAULT_LAGS = [0, 1, 2, 5]
_MIN_N = 200                      # bir hücrenin raporlanması için asgari örneklem


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
def load_predictions(path: str | None = None) -> pd.DataFrame:
    """predictions_log.csv → (symbol, key, sign, confidence, price, ...) çerçevesi.

    `key` = `_canon_ts` ile kanonikleştirilmiş bar zaman damgası (üretimdeki eşleme
    sözleşmesinin AYNISI — tz-aware kayıt / naive çözüm karışımı buradan gelir)."""
    p = path or str(DATA_DIR / "predictions_log.csv")
    df = pd.read_csv(p)
    df = df[df["direction"].isin(["+", "-"])].copy()
    df["sign"] = np.where(df["direction"].to_numpy() == "+", 1.0, -1.0)
    df["key"] = [_canon_ts(t) for t in df["bar_ts"]]
    df["ts"] = pd.to_datetime(df["key"], errors="coerce")
    df = df[df["ts"].notna()]
    for c in ("confidence", "price", "hit", "pwin", "prob_cal"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def to_rth_grid(df: pd.DataFrame) -> pd.DataFrame:
    """SIP barlarını KESİNTİSİZ seans-içi (RTH) dakika ızgarasına indir.

    NEDEN: ham SIP serisi uzatılmış seans barlarını da içerir ve işlem görmeyen dakikada
    bar YOKTUR (AAPL'de RTH kapsaması %95,7). O seride "H bar ileri" ne sabit bir dakika
    sayısıdır ne de seans içinde kalır — H=390 kimi zaman gece boşluğuna düşer. Izgara
    09:30–15:59 ET dakikalarına kırpılıp eksik dakikalar son kapanışla doldurulunca
    "H bar" = "H seans-dakikası" olur (seanslar ardışık birleştirilir; gece boşluğu
    yalnız H seans sonunu aşarsa devreye girer ve `ayni_seans_orani` ile raporlanır)."""
    if df is None or df.empty:
        return df
    idx = pd.to_datetime(pd.Series(df.index))
    et = idx.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    mins = et.dt.hour * 60 + et.dt.minute
    keep = (mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 59)
    d = df[keep.to_numpy()]
    if d.empty:
        return d
    et_k = et[keep.to_numpy()]
    parts = []
    for day, sub in d.groupby(et_k.dt.date.to_numpy()):
        first = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(minutes=9 * 60 + 30)
        full = pd.date_range(first, periods=390, freq="1min").tz_convert("UTC").tz_localize(None)
        parts.append(sub.reindex(full).ffill())
    out = pd.concat(parts)
    return out.dropna(subset=["close"])


def fetch_bars(symbols: list[str], start: datetime, end: datetime, interval: str = "1m",
               cache_days: int = 30, grid: str = "rth") -> dict[str, pd.DataFrame]:
    """Sembol → 1m bar çerçevesi (alpaca_history, disk-cache'li).

    `cache_days`: araştırma koşusu gece yarısını geçince cache'in bayatlayıp 36 sembolü
    yeniden çekmesini engeller (alpaca_history._read_cache tazelik kapısı).
    `grid="rth"` → `to_rth_grid` (varsayılan, ufuk = seans-dakikası); `"raw"` → ham SIP."""
    os.environ.setdefault("ALPACA_HIST_CACHE_MAX_AGE_DAYS", str(cache_days))
    from nihai.data.alpaca_history import get_history
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = get_history(sym, interval, start=start, end=end)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 — tek sembol düşerse süpürme sürsün
            print(f"[sweep] {sym} çekilemedi: {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            print(f"[sweep] {sym}: bar yok", file=sys.stderr)
            continue
        ham = len(df)
        if grid == "rth":
            df = to_rth_grid(df)
            if df.empty:
                continue
        out[sym] = df
        print(f"[sweep] {i}/{len(symbols)} {sym}: {ham} ham → {len(df)} {grid} bar "
              f"({df.index[0]} → {df.index[-1]})", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# İstatistik
# ---------------------------------------------------------------------------
def _cluster_t(signed: np.ndarray, clusters: np.ndarray) -> tuple[float | None, int]:
    """Küme-ortalamaları üzerinden t (üst üste binen pencerelerin şişirdiği t'ye panzehir).

    1m frekansta H barlık ileri getiriler neredeyse tamamen örtüşür → gözlemler bağımsız
    DEĞİL. Sembol×gün kümesinin ortalamasını tek gözlem sayarak etkin örneklem küçültülür."""
    if signed.size == 0:
        return None, 0
    s = pd.Series(signed).groupby(pd.Series(clusters)).mean()
    k = int(s.size)
    if k < 3:
        return None, k
    sd = float(s.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None, k
    return float(s.mean() / (sd / np.sqrt(k))), k


def _stats(signed: np.ndarray, net: np.ndarray, clusters: np.ndarray | None = None) -> dict:
    """Bir hücrenin karnesi. `isabet` işaretli getirinin pozitifliği (maliyetsiz),
    kârlılık ölçüleri NET (maliyet düşülmüş) getiriden."""
    signed = np.asarray(signed, dtype=float)
    net = np.asarray(net, dtype=float)
    ok = np.isfinite(signed) & np.isfinite(net)
    signed, net = signed[ok], net[ok]
    if signed.size == 0:
        return {"n": 0}
    wins, losses = net[net > 0], net[net < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else None
    mw = float(wins.mean()) if wins.size else 0.0
    ml = float(abs(losses.mean())) if losses.size else 0.0
    rr = (mw / ml) if ml > 0 else None
    sd = float(signed.std(ddof=1)) if signed.size > 1 else 0.0
    out: dict[str, Any] = {
        "n": int(signed.size),
        "isabet": round(float((signed > 0).mean()), 4),
        "ort_getiri_pct": round(float(signed.mean()) * 100, 5),
        "ort_net_getiri_pct": round(float(net.mean()) * 100, 5),
        "profit_factor_net": round(pf, 4) if pf is not None else None,
        "odul_risk": round(rr, 4) if rr is not None else None,
        "basabas_isabet": round(1.0 / (1.0 + rr), 4) if rr else None,
        "t_ham": round(float(signed.mean() / (sd / np.sqrt(signed.size))), 3) if sd > 0 else None,
    }
    if clusters is not None and clusters.size == ok.size:
        tc, k = _cluster_t(net, clusters[ok])
        out["t_kume"] = round(tc, 3) if tc is not None else None
        out["n_kume"] = k
    return out


def _sidak_t(m: int, alpha: float = 0.05) -> float:
    """m bağımsız denemede aile-bazlı α için kritik |t| (normal yaklaşımı).

    Gerçek denemeler bağımsız değil (aynı veri, örtüşen pencereler) → bu eşik
    MUHAFAZAKÂR DEĞİL, asgari bir kapıdır. Geçmek gerek şarttır, yeter şart değildir."""
    from scipy.stats import norm  # scipy zaten bağımlılık (research/*)
    p = 1.0 - (1.0 - alpha) ** (1.0 / max(1, m))
    return float(abs(norm.ppf(p / 2.0)))


# ---------------------------------------------------------------------------
# Süpürme
# ---------------------------------------------------------------------------
def sweep(preds: pd.DataFrame, bars: dict[str, pd.DataFrame], horizons: list[int],
          lags: list[int], cost_frac: dict[str, float], same_session_only: bool = False,
          ) -> tuple[pd.DataFrame, dict]:
    """Her tahmini her (H, lag) için yeniden puanla → uzun-format sonuç çerçevesi.

    Look-ahead yok: giriş `bar_ts + lag` barının KAPANIŞI, çıkış `giriş + H` barının
    kapanışı. Karar barının kendi kapanışı girişte kullanılabilir çünkü tahmin zaten o
    bar KAPANDIKTAN sonra loglanmıştır (lag=0 = 'kapanışta gir' varsayımı; lag≥1 gerçekçi)."""
    recs: list[dict] = []
    diag: dict[str, Any] = {"n_tahmin": int(len(preds)), "eslesen": 0, "eslesmeyen": 0,
                            "hizalama_bps": [], "kayma_bps": [], "sembol_yok": 0}

    for sym, grp in preds.groupby("symbol"):
        df = bars.get(str(sym))
        if df is None or df.empty:
            diag["sembol_yok"] += int(len(grp))
            continue
        closes = df["close"].to_numpy(dtype=float)
        idx = df.index
        pos_map: dict[str, int] = {}
        for j, t in enumerate(idx):
            k = _canon_ts(t)
            if k not in pos_map:
                pos_map[k] = j
        n = closes.size
        days = pd.Series(pd.to_datetime(pd.Series(idx)).dt.date.to_numpy())

        pos = np.array([pos_map.get(k, -1) for k in grp["key"]], dtype=int)
        hit_mask = pos >= 0
        diag["eslesen"] += int(hit_mask.sum())
        diag["eslesmeyen"] += int((~hit_mask).sum())
        if not hit_mask.any():
            continue
        g = grp[hit_mask].reset_index(drop=True)
        pos = pos[hit_mask]

        # Hizalama kanıtı: logdaki fiyat ile karar barı kapanışı örtüşüyor mu?
        pr = g["price"].to_numpy(dtype=float)
        bar_close = closes[pos]
        with np.errstate(invalid="ignore", divide="ignore"):
            dev = np.abs(pr / bar_close - 1.0) * 1e4
        diag["hizalama_bps"].extend(dev[np.isfinite(dev)].tolist())

        sgn = g["sign"].to_numpy(dtype=float)
        conf = g["confidence"].to_numpy(dtype=float)
        logged_hit = (g["hit"].to_numpy(dtype=float) if "hit" in g.columns
                      else np.full(pos.size, np.nan))
        cost = float(cost_frac.get(str(sym), cost_frac.get("_default", 0.0)))
        ts_arr = g["ts"].to_numpy()

        # GİRİŞ KAYMASI (işaretli): canlı `price`, karar barı kapanışından sinyal YÖNÜNDE
        # ne kadar uzakta? Pozitif = pahalı girilmiş (ters seçilim / gecikme maliyeti).
        with np.errstate(invalid="ignore", divide="ignore"):
            slip = (pr / bar_close - 1.0) * 1e4 * sgn
        diag["kayma_bps"].extend(slip[np.isfinite(slip)].tolist())

        for lag in lags:
            e_idx = pos + lag
            for h in horizons:
                x_idx = e_idx + h
                ok = (e_idx >= 0) & (x_idx < n)
                if not ok.any():
                    continue
                ei, xi = e_idx[ok], x_idx[ok]
                exit_ = closes[xi]
                same = (days.to_numpy()[ei] == days.to_numpy()[xi])
                sub_all = np.flatnonzero(ok)
                # İki giriş varsayımı: "bar" = karar/gecikme barının KAPANIŞI (kâğıt üstü),
                # "canli" = sinyal anında loglanan gerçek fiyat (yalnız lag=0'da anlamlı;
                # canlı fiyat o ana aittir). Aradaki fark = ölçülen giriş kayması.
                variants = [("bar", closes[ei])]
                if lag == 0:
                    variants.append(("canli", pr[sub_all]))
                for gname, entry in variants:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        fwd = exit_ / entry - 1.0
                    good = np.isfinite(fwd) & (entry > 0)
                    if same_session_only:
                        good = good & same
                    if not good.any():
                        continue
                    sub = sub_all[good]
                    recs.append({
                        "lag": lag, "H": h, "symbol": str(sym), "giris": gname,
                        "signed": (fwd[good] * sgn[sub]),
                        "cost": cost, "conf": conf[sub], "ts": ts_arr[sub],
                        "same": same[good], "logged_hit": logged_hit[sub],
                    })
    if not recs:
        return pd.DataFrame(), diag

    # Uzun format: her satır tek bir yeniden-puanlama
    frames = []
    for r in recs:
        frames.append(pd.DataFrame({
            "lag": r["lag"], "H": r["H"], "symbol": r["symbol"], "giris": r["giris"],
            "signed": r["signed"], "cost": r["cost"], "conf": r["conf"],
            "ts": r["ts"], "same": r["same"], "logged_hit": r["logged_hit"],
        }))
    long = pd.concat(frames, ignore_index=True)
    for c in ("symbol", "giris"):
        long[c] = long[c].astype("category")            # 1,4M satırda str kolonu ~100MB
    long["net"] = long["signed"] - long["cost"]
    dev_arr = np.asarray(diag.pop("hizalama_bps"), dtype=float)
    diag["hizalama_medyan_bps"] = round(float(np.median(dev_arr)), 3) if dev_arr.size else None
    diag["hizalama_p95_bps"] = (round(float(np.percentile(dev_arr, 95)), 3)
                                if dev_arr.size else None)
    slip_arr = np.asarray(diag.pop("kayma_bps"), dtype=float)
    if slip_arr.size:
        sd = float(slip_arr.std(ddof=1)) if slip_arr.size > 1 else 0.0
        diag["giris_kaymasi"] = {
            "ort_bps": round(float(slip_arr.mean()), 3),
            "medyan_bps": round(float(np.median(slip_arr)), 3),
            "t": round(float(slip_arr.mean() / (sd / np.sqrt(slip_arr.size))), 2) if sd > 0 else None,
            "n": int(slip_arr.size),
            "NOT": ("Pozitif = canlı giriş fiyatı sinyal YÖNÜNDE bar kapanışından uzaklaşmış "
                    "(pahalı girilmiş). Bu, tahmin hatası DEĞİL uygulama maliyetidir; canlı "
                    "`hit` kolonu bu maliyeti isabet düşüşü olarak gösterir."),
        }
    return long, diag


def summarize(long_all: pd.DataFrame, horizons: list[int], lags: list[int],
              min_n: int = _MIN_N, giris: str = "bar") -> dict:
    """(H, lag) hücrelerinin karnesi + zaman-bölmeli doğrulama + verdikt.

    `giris="bar"` kâğıt-üstü (bar kapanışında gir), `"canli"` gerçekleşen giriş fiyatı."""
    long = long_all[long_all["giris"] == giris]
    if long.empty:
        return {"hata": f"yeniden puanlanan satır yok (giris={giris})"}
    ts = pd.to_datetime(long["ts"])
    mid = ts.min() + (ts.max() - ts.min()) / 2
    clusters = (long["symbol"].astype(str) + "|" + ts.dt.date.astype(str)).to_numpy()

    cells: dict[str, Any] = {}
    m = max(1, len([1 for _ in horizons for _ in lags]))
    t_crit = _sidak_t(m)
    aday: list[dict] = []

    for lag in lags:
        for h in horizons:
            sel = (long["lag"] == lag) & (long["H"] == h)
            sub = long[sel]
            if len(sub) < min_n:
                continue
            key = f"H{h}_lag{lag}"
            st = _stats(sub["signed"].to_numpy(), sub["net"].to_numpy(), clusters[sel.to_numpy()])
            half1 = sub[ts[sel] <= mid]
            half2 = sub[ts[sel] > mid]
            st["ilk_yari_net_pct"] = (round(float(half1["net"].mean()) * 100, 5)
                                      if len(half1) else None)
            st["ikinci_yari_net_pct"] = (round(float(half2["net"].mean()) * 100, 5)
                                         if len(half2) else None)
            st["ayni_seans_orani"] = round(float(sub["same"].mean()), 3)
            cells[key] = st
            if (lag >= 1 and (st["ort_net_getiri_pct"] or 0) > 0
                    and (st.get("t_kume") or 0) > t_crit
                    and (st["ilk_yari_net_pct"] or 0) > 0
                    and (st["ikinci_yari_net_pct"] or 0) > 0):
                aday.append({"hucre": key, **{k: st[k] for k in
                                              ("n", "ort_net_getiri_pct", "t_kume")}})

    out: dict[str, Any] = {
        "hucreler": cells,
        "sidak_kritik_t": round(t_crit, 3),
        "denenen_kombinasyon": m,
        "KABUL_ADAYLARI": aday,
        "VERDIKT": ("H* BULUNDU (aday hücreler kabul kriterini geçti — yine de OOS/ileri "
                    "doğrulama gerekir)" if aday else
                    "H* YOK — hiçbir ufuk/gecikme kombinasyonu maliyet sonrası, kümelenmiş t "
                    "ve iki-yarı kapılarını birlikte geçmedi (EDGE_FINDINGS §23 adayı)"),
    }
    return out


def confidence_cross(long: pd.DataFrame, horizons: list[int], lag: int = 1,
                     q: int = 5) -> dict:
    """Güven dilimi × H çapraz tablosu (F2'nin girdisi).

    Denetimde 5 dilimde isabet ile güven arasında MONOTON TERS ilişki ölçüldü (ρ=−0,927).
    Bu tablo, o ilişkinin ufka bağlı olup olmadığını gösterir."""
    sub = long[(long["lag"] == lag) & (long["giris"] == "bar") & long["conf"].notna()]
    if sub.empty:
        return {}
    try:
        dilim = pd.qcut(sub["conf"], q, labels=False, duplicates="drop")
    except ValueError:
        return {}
    out: dict[str, Any] = {}
    for h in horizons:
        s = sub[sub["H"] == h]
        if s.empty:
            continue
        d = dilim[sub["H"] == h]
        g = s.groupby(d.to_numpy())
        out[f"H{h}"] = {
            "isabet": {int(k): round(float(v), 4) for k, v in (s["signed"] > 0).groupby(
                d.to_numpy()).mean().items()},
            "net_pct": {int(k): round(float(v) * 100, 5) for k, v in
                        g["net"].mean().items()},
            "n": {int(k): int(v) for k, v in g.size().items()},
        }
        # dilim sırası ↔ isabet korelasyonu (monotonluk göstergesi)
        hits = (s["signed"] > 0).groupby(d.to_numpy()).mean()
        if hits.size >= 3:
            out[f"H{h}"]["dilim_isabet_korelasyonu"] = round(
                float(np.corrcoef(np.asarray(hits.index, dtype=float),
                                  hits.to_numpy())[0, 1]), 4)
    return out


def parity_check(long: pd.DataFrame) -> dict:
    """H=5 / lag=0 hücresinde logdaki `hit` yeniden üretilebiliyor mu?

    Bu ölçümün ÖN KOŞULU: parite düşükse bulgular sinyal hakkında değil, bar HİZALAMASI
    hakkındadır. Üretimdeki `hit` = işaret(fwd_price − price); `price` CANLI fiyat,
    `fwd_price` +5 barın KAPANIŞI → parite `giris="canli"` varyantıyla ölçülür. `giris="bar"`
    aynı tahminlerin kâğıt üstünde (karar barı kapanışında girilseydi) ne vereceğidir; ikisi
    arasındaki isabet farkı doğrudan GİRİŞ KAYMASININ bedelidir."""
    out: dict[str, Any] = {}
    for gname in ("canli", "bar"):
        sub = long[(long["H"] == 5) & (long["lag"] == 0) & (long["giris"] == gname)
                   & long["logged_hit"].notna()]
        if sub.empty:
            continue
        yeniden = (sub["signed"] > 0).astype(float).to_numpy()
        kayitli = sub["logged_hit"].to_numpy(dtype=float)
        out[gname] = {"n": int(len(sub)), "uyum": round(float((yeniden == kayitli).mean()), 4),
                      "kayitli_isabet": round(float(kayitli.mean()), 4),
                      "yeniden_isabet": round(float(yeniden.mean()), 4)}
    if "canli" in out and "bar" in out:
        out["kayma_bedeli_isabet_puani"] = round(
            out["bar"]["yeniden_isabet"] - out["canli"]["yeniden_isabet"], 4)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(horizons: list[int] | None = None, lags: list[int] | None = None,
        spread_model: str = "none", cost_bps: float = 0.0, log_path: str | None = None,
        max_symbols: int = 0, same_session_only: bool = False,
        cache_days: int = 30, pad_days: int = 3, grid: str = "rth") -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    lags = lags or DEFAULT_LAGS
    preds = load_predictions(log_path)
    if preds.empty:
        return {"hata": "tahmin logu boş"}
    syms = sorted(preds["symbol"].astype(str).unique())
    if max_symbols:
        syms = syms[:max_symbols]
        preds = preds[preds["symbol"].astype(str).isin(syms)]

    start = preds["ts"].min().to_pydatetime().replace(tzinfo=UTC) - timedelta(days=1)
    # En büyük ufuk + gecikme kadar İLERİ bar gerekir; takvim payıyla genişlet.
    end = (preds["ts"].max().to_pydatetime().replace(tzinfo=UTC)
           + timedelta(days=max(pad_days, int(max(horizons) / 390) + 2)))
    bars = fetch_bars(syms, start, end, "1m", cache_days, grid)
    if not bars:
        return {"hata": "hiç bar çekilemedi (ALPACA anahtarı? SIP planı?)"}

    cost_frac: dict[str, float] = {"_default": 2.0 * cost_bps / 1e4}
    spread_tbl: dict[str, dict[str, float]] = {}
    if spread_model in ("corwin", "roll", "max"):
        from research.spread_estimate import estimate
        for sym, df in bars.items():
            est = estimate(df, 1.0)
            spread_tbl[sym] = est
            # Gidiş-dönüş maliyeti ≈ bir TAM etkin spread (her bacakta yarım spread aşılır).
            cost_frac[sym] = float(est.get(f"{spread_model}_bps", 0.0)) / 1e4
    elif spread_model not in ("none", "fixed_bps"):
        return {"hata": f"bilinmeyen spread modeli: {spread_model}"}

    long, diag = sweep(preds, bars, horizons, lags, cost_frac, same_session_only)
    if long.empty:
        return {"hata": "hiç tahmin bara eşlenemedi", "teshis": diag}

    ozet = summarize(long, horizons, lags, giris="bar")
    canli = summarize(long, horizons, lags, giris="canli")
    out: dict[str, Any] = {
        "kapsam": {**diag, "sembol": len(bars), "ufuklar": horizons, "gecikmeler": lags,
                   "izgara": grid, "ayni_seans_zorunlu": same_session_only},
        "maliyet": {"model": spread_model, "cost_bps_sabit": cost_bps,
                    "medyan_spread_bps": (round(float(np.median(
                        [v.get(f"{spread_model}_bps", 0.0) for v in spread_tbl.values()])), 3)
                        if spread_tbl else None)},
        "parite_H5_lag0": parity_check(long),
        **ozet,
        "canli_giris_hucreleri": canli.get("hucreler", {}),
        "guven_dilimi_capraz": confidence_cross(long, horizons, lag=min(lags) if 1 not in lags else 1),
        "UYARI": ("Bu bir KEŞİF süpürmesidir: aynı veri üzerinde çok sayıda kombinasyon "
                  "denenir. Kabul adayı çıksa bile bu OOS kanıtı DEĞİLDİR — ileri doğrulama "
                  "(prediction_tracker) veya purged CV şarttır (ACI DERS #3/#4)."),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Canlı tahminlerin ufuk süpürmesi (geriye dönük)")
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--exec-lags", default=",".join(str(x) for x in DEFAULT_LAGS))
    ap.add_argument("--spread-model", default="none",
                    choices=["none", "corwin", "roll", "max", "fixed_bps"])
    ap.add_argument("--cost-bps", type=float, default=0.0, help="fixed_bps modelinde TEK bacak")
    ap.add_argument("--log", default=None, help="predictions_log.csv yolu")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--same-session", action="store_true", help="giriş/çıkış aynı seansta olsun")
    ap.add_argument("--grid", default="rth", choices=["rth", "raw"],
                    help="rth = kesintisiz seans-dakikası ızgarası (varsayılan)")
    ap.add_argument("--cache-days", type=int, default=30)
    ap.add_argument("--json", default=None, help="sonucu bu dosyaya da yaz")
    a = ap.parse_args(argv)
    hs = [int(x) for x in a.horizons.split(",") if x.strip()]
    lg = [int(x) for x in a.exec_lags.split(",") if x.strip()]
    res = run(horizons=hs, lags=lg, spread_model=a.spread_model, cost_bps=a.cost_bps,
              log_path=a.log, max_symbols=a.max_symbols, same_session_only=a.same_session,
              cache_days=a.cache_days, grid=a.grid)
    txt = json.dumps(res, ensure_ascii=False, indent=2, default=str)
    print(txt)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
