"""
build_hist_1m_dataset.py — canlı 1m tahmin satırlarının TARİHSEL eşi (Tur-94 pilotu).

Canlı eğitim seti (`build_live_dataset`) yalnız servisin koştuğu günlerden birikir (Tur-93: 36 gün,
16.198 satır — 282M basamağının kendi alt sınırının 3,6 milyon olduğu yerde). Bu modül AYNI satırı
tarihsel IEX 1m barlarından, CANLI YOLUN KENDİ FONKSİYONLARIYLA yeniden kurar:

  pencere : RTH süzgeçli (Tur-90 `rth_mask`), gün-içi `split` düzeltmeli (Tur-91), son 220 bar
            (`live_predict_service` `lookback_bars=220`) — göstergeler pencereye bağlıdır (EMA başlangıcı),
            tam seriden hesaplamak canlıdan farklı bir dağılım üretirdi (Tur-20 dersi).
  aday    : `generate_signal(sym, pencere, shared, live_interval="1m")` → `_candidate_side`
  özellik : `extract_features(pencere, sig, shared)`; `is_long` aday tarafıdır (canlıyla aynı düzeltme)
  etiket  : aday tarafına göre işaretli getiri > 0; hedef an t0 + 5 dk (`prediction_tracker._forward_price`,
            Tur-84 zaman kuralı; seans sonunda kapanış fiyatı); çapa `--capa` (vars. `orta_i1`, bkz.
            `taraf_getirisi`: bar kapanışı çapası makas sıçramasını öğretir)
Bilinçli farklar: referans fiyat sonraki barın ortası (canlıda ~39 sn sonraki kotasyon) ve her
`adim`. bar (vars. 5 = ufuk → aynı sembolde etiketler örtüşmez; tekdüze alt örneklem aday popülasyonunu
bozmaz). `--yeniden-etiketle` var olan setin etiketini özellikleri yeniden hesaplamadan kurar.
SINIR: bugünün takip listesi geçmişe uygulanır → survivorship (ACI DERS #1) — sonuç okunurken hesaba katılır.

Kullanım (repo kökünden; barlar `ALPACA_HIST_DIR` önbelleğinde, `{SYM}_1m_split_iex.csv`):
    python -m research.build_hist_1m_dataset --hist-dir C:\\Users\\Kagan\\nihai_hist\\alpaca_1m_iex \
        --out C:\\Users\\Kagan\\nihai_hist\\meta_dataset_hist_1m_2y.csv --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from nihai.config import DATA_DIR
from nihai.console import ensure_utf8_stdout
from nihai.signals.meta_features import FEATURE_NAMES, neutral_default

EXTRA_COLS = ["pnl", "label", "symbol", "period", "entry_time", "bars_held",
              "mae_pct", "mfe_pct", "removed_since"]
PENCERE = 220   # live_predict_service: provider.get_bars(symbol, "1m", lookback_bars=220)
UFUK = 5        # prediction_tracker HORIZON_BARS


def rth_barlar(df: pd.DataFrame) -> pd.DataFrame:
    """Canlı 1m barıyla aynı küme: yalnız NYSE normal seansı (Tur-90), zaman sıralı, tekil."""
    from nihai.data.data_provider import rth_mask
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[rth_mask(df.index, 1)]


def taraf_getirisi(df: pd.DataFrame, pos_map: dict, i: int, side: str, *, capa: str = "orta_i1",
                   ufuk: int = UFUK) -> float | None:
    """Karar barı i'deki adayın taraf-işaretli getirisi (%). Hedef canlıyla aynı (t0 + ufuk dk).

    ÇAPA (Tur-94): canlı etiket bar kapanışından ~39 sn SONRA okunan kotasyona çapalıdır. Eşleşen
    14.428 canlı satırda uyum: `kapanis_i` %75,2 · `orta_i1` (sonraki barın açılış-kapanış ortası)
    %76,1. `kapanis_i` ayrıca makas sıçramasını taşır: diğer sembollerde AUC 0,5215 → 1 bar
    gecikmede 0,5004 (ACI DERS #7) — model artefaktı öğrenir. Varsayılan bu yüzden `orta_i1`;
    sonraki bar seans dışındaysa (≥ ufuk dk boşluk) satır yok."""
    from nihai.execution.prediction_tracker import _forward_price
    fwd = _forward_price(df, pos_map, str(df.index[i]), ufuk, "1m")
    if capa == "kapanis_i":
        c0 = float(df["close"].iat[i])
    elif capa == "orta_i1":
        if i + 1 >= len(df) or df.index[i + 1] - df.index[i] >= pd.Timedelta(minutes=ufuk):
            return None
        c0 = (float(df["open"].iat[i + 1]) + float(df["close"].iat[i + 1])) / 2.0
    else:
        raise ValueError(f"bilinmeyen çapa: {capa!r} (kapanis_i|orta_i1)")
    if fwd is None or not np.isfinite(c0) or c0 <= 0:
        return None
    ham = (fwd / c0 - 1.0) * 100.0
    return ham if side == "LONG" else -ham


def sembol_satirlari(sym: str, df: pd.DataFrame, *, adim: int = 5, pencere: int = PENCERE,
                     ufuk: int = UFUK, capa: str = "orta_i1", strateji: str | None = None) -> list[dict]:
    """Tek sembolün tarihsel aday satırları. `df` RTH süzülmüş 1m barlardır.

    `strateji` None = yürürlükteki canlı strateji (eski davranış). Tur-107: canlı strateji `quant_trend`e
    geçince (Tur-97) 1m barda aday kalmadı ve bu kurucu sessizce 0 satır üretti; geçmiş bir modelin aday
    popülasyonunu yeniden kurmak isteyen çağıran stratejiyi AÇIKÇA verir."""
    from nihai.execution.live_predict_service import _candidate_side
    from nihai.execution.prediction_tracker import _pos_map
    from nihai.signals.meta_features import extract_features
    from nihai.signals.strategy import _SharedIndicators, generate_signal

    rows: list[dict] = []
    if len(df) <= pencere:
        return rows
    pos_map = _pos_map(df)
    for i in range(pencere - 1, len(df) - 1, max(1, int(adim))):
        w = df.iloc[i - pencere + 1:i + 1]
        sh = _SharedIndicators(w)
        sig = generate_signal(sym, w, shared=sh, live_interval="1m", strategy_mode=strateji)
        side = _candidate_side(sig)
        if side is None:
            continue
        ret = taraf_getirisi(df, pos_map, i, side, capa=capa, ufuk=ufuk)
        if ret is None:
            continue
        f = dict(extract_features(w, sig, shared=sh))
        f["is_long"] = 1.0 if side == "LONG" else 0.0
        out: dict[str, object] = {}
        for n in FEATURE_NAMES:
            v = f.get(n)
            try:
                v = float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                v = float("nan")
            out[n] = v if np.isfinite(v) else neutral_default(n)
        out.update({"pnl": ret, "label": 1 if ret > 0 else 0, "symbol": sym, "period": "hist",
                    "entry_time": str(df.index[i]), "bars_held": ufuk,
                    "mae_pct": "", "mfe_pct": "", "removed_since": ""})
        rows.append(out)
    return rows


def _isci(sym: str, hist_dir: str, parca_dir: str, adim: int, capa: str) -> tuple[str, int, float]:
    t = time.time()
    hedef = Path(parca_dir) / f"{sym}.csv"
    if hedef.exists():
        return sym, -1, 0.0                     # önceki koşudan kalan parça (devam)
    kaynak = Path(hist_dir) / f"{sym.replace('/', '_').replace('.', '-')}_1m_split_iex.csv"
    if not kaynak.exists():
        return sym, 0, 0.0
    df = pd.read_csv(kaynak, index_col=0, parse_dates=True)
    rows = sembol_satirlari(sym, rth_barlar(df), adim=adim, capa=capa)
    tmp = hedef.with_suffix(".tmp")
    pd.DataFrame(rows, columns=[*FEATURE_NAMES, *EXTRA_COLS]).to_csv(tmp, index=False)
    tmp.replace(hedef)
    return sym, len(rows), time.time() - t


def _etiketle_isci(sym: str, zamanlar: list[str], uzunlar: list[float], hist_dir: str,
                   capa: str) -> tuple[str, list[float]]:
    """Var olan satırların etiketini (özellik yeniden hesaplanmadan) başka çapayla kur."""
    from nihai.execution.prediction_tracker import _canon_ts, _pos_map
    kaynak = Path(hist_dir) / f"{sym.replace('/', '_').replace('.', '-')}_1m_split_iex.csv"
    df = rth_barlar(pd.read_csv(kaynak, index_col=0, parse_dates=True))
    pm = _pos_map(df)
    out: list[float] = []
    for z, u in zip(zamanlar, uzunlar, strict=True):
        i = pm.get(_canon_ts(z))
        r = None if i is None else taraf_getirisi(df, pm, i, "LONG" if u > 0.5 else "SHORT", capa=capa)
        out.append(np.nan if r is None else r)
    return sym, out


def yeniden_etiketle(kaynak: str, out: str, hist_dir: str, capa: str, workers: int) -> int:
    df = pd.read_csv(kaynak)
    pnl = np.full(len(df), np.nan)
    gruplar = df.groupby("symbol").indices
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        isler = {ex.submit(_etiketle_isci, s, df["entry_time"].iloc[ix].tolist(),
                           df["is_long"].iloc[ix].tolist(), hist_dir, capa): ix
                 for s, ix in gruplar.items()}
        for k, f in enumerate(as_completed(isler), 1):
            sym, r = f.result()
            pnl[isler[f]] = r
            print(f"{k}/{len(gruplar)} {sym}", flush=True)
    df["pnl"] = pnl
    df["label"] = (pnl > 0).astype(int)
    df = df[np.isfinite(pnl)]
    df.to_csv(out, index=False)
    print(f"→ {out}: {len(df):,} satır (çapa {capa}) / taban %{100 * df['label'].mean():.1f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tarihsel 1m aday satırları (canlı yolun eşi)")
    ap.add_argument("--hist-dir", required=True, help="{SYM}_1m_split_iex.csv önbelleği")
    ap.add_argument("--out", required=True, help="birleşik CSV (ad '_1m_' içermeli: kart aralığı)")
    ap.add_argument("--symbols", default=None, help="virgüllü; verilmezse data/watchlist.json")
    ap.add_argument("--adim", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--capa", default="orta_i1", choices=["orta_i1", "kapanis_i"])
    ap.add_argument("--yeniden-etiketle", default=None, metavar="KAYNAK_CSV",
                    help="özellikleri yeniden hesaplamadan var olan setin etiketini --capa ile kur")
    args = ap.parse_args(argv)

    if "_1m_" not in Path(args.out).name:
        print("HATA: çıktı adı '_1m_' içermeli (neural_shadow kart aralığını dosya adından okur).")
        return 1
    if args.yeniden_etiketle:
        return yeniden_etiketle(args.yeniden_etiketle, args.out, args.hist_dir, args.capa, args.workers)
    if args.symbols:
        semboller = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        semboller = json.loads((DATA_DIR / "watchlist.json").read_text(encoding="utf-8"))["symbols"]
    parca = Path(args.out).with_suffix("")
    parca = parca.parent / f"{parca.name}_parcalar"
    parca.mkdir(parents=True, exist_ok=True)
    print(f"sembol={len(semboller)} adım={args.adim} pencere={PENCERE} ufuk={UFUK} dk "
          f"işçi={args.workers} parça={parca}", flush=True)
    toplam = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        isler = [ex.submit(_isci, s, args.hist_dir, str(parca), args.adim, args.capa) for s in semboller]
        for k, f in enumerate(as_completed(isler), 1):
            sym, n, dt = f.result()
            toplam += max(n, 0)
            print(f"{k}/{len(semboller)} {sym} {'devam' if n < 0 else f'{n} satır {dt:.0f} sn'} "
                  f"(toplam {toplam:,})", flush=True)
    parcalar = sorted(parca.glob("*.csv"))
    birlesik = pd.concat([pd.read_csv(p) for p in parcalar], ignore_index=True)
    birlesik.to_csv(args.out, index=False)
    print(f"→ {args.out}: {len(birlesik):,} satır / {birlesik['symbol'].nunique()} sembol / "
          f"taban %{100 * birlesik['label'].mean():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
