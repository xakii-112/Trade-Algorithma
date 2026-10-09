"""noral_282m_backtest.py — canlı gölge 282M sinir ağının 10 yıllık backtest'i ve kıyası (Tur-107).

KENAR İDDİASI YOK. Model yeniden EĞİTİLMEZ; kayıtlı canlı artefakt (Tur-93, `quant_transformer_moe_xl`,
kart `c70acd68…`) olduğu gibi yüklenir ve hiç görmediği tarihsel aday satırlarında sınanır. Canlıya,
karta, `.env`e ve kayıt defterine dokunulmaz; çıktılar `data/noral_282m_*.json` + OneDrive DIŞI ara dosyalar.

ÖN-KAYIT (sonuç görülmeden yazıldı, 2026-10-06):
- Evren: modelin eğitim evreni olan 30 sembol (bugünün listesi → survivorship; rapora yazılır).
- Pencere: 2016-10-06 → 2026-10-05. Model 1 dakikalık özelliklerle eğitildi; 10 yıl geriye giden tek 1m
  kaynağı Alpaca SIP'tir (IEX 1m 2020 ortasında başlar — ölçüldü). SIP ↔ IEX besleme farkı, iki beslemenin
  de bulunduğu son iki yılda AYNI semboller üzerinde ayrıca raporlanır.
- Satırlar: canlı yolun kendi fonksiyonları (`research.build_hist_1m_dataset`): seans içi, son 220 bar,
  `generate_signal` → aday tarafı → `extract_features`; etiket = 5 dk taraf getirisi, çapa sonraki barın ortası.
- Dönemler ayrı raporlanır: EĞİTİM ÖNCESİ (< 2026-07-06, model hiç görmedi) · EĞİTİM DÖNEMİ (2026-07-06 →
  2026-09-23, örneklem-içi, hükme girmez) · İLERİ (> 2026-09-23).
- Ölçüler: dağılım-içi pay (canlı OOD kapısı) · sembol-içi AUC · Brier ↔ SABİT taban (yalnız geçmişten,
  genişleyen) + gün-kümeli eşleştirilmiş t · çıktı yayılımı (std) · kapı simülasyonu (canlı gölge kasanın
  kuralı: p ≥ son 200 ölçümün medyanı; tutulan − taban, gün-kümeli t) · aynı oranda RASTGELE kapı · +1 bar
  gecikme · günlük eşit ağırlıklı getiri serisi ↔ SPY al-tut.
- Kıyas modelleri: sabit taban oran · önceki canlı model `deep_mlp` (186 bin parametre) · rastgele kapı.
- "Beceri var" denebilmesi için HEPSİ gerekir: eğitim öncesi dönemde Brier farkının gün-kümeli t < −1,645,
  kapı fazlalığının gün-kümeli t > 1,645, +1 bar gecikmede işaretin korunması. Aksi hâlde hüküm "beceri yok".

EK (2026-10-06, model çıktısı görülmeden, artefakt manifesti okununca): eğitimde sabit olduğu için nötrlenen
iki kolon (`rr_ratio`, `confluence`; manifestte μ=0, σ=1) canlı yolda HAM beslenir (z=2,0 ve z=1,0). Her
ölçü iki besleme kipinde raporlanır: `canli` (ham; canlı gölge yolun yaptığı) ve `egitim` (iki kolon 0).
Hüküm `canli` kipinden okunur — soru "projedeki AI ne yapıyor"dur.

EK-2 (2026-10-06, KIYAS çıktısı görüldükten SONRA, backtest çıktısı görülmeden): dropout kapalı `predict_proba`
bu artefaktta girdiden bağımsız tek bir sayı veriyor (300 canlı satırda yayılım 1,3e-7); canlı köprünün yazdığı
değer ise MC-dropout ortalamasıdır (yayılım 5,3e-4, eval ile korelasyon 0,19). Eval kipiyle ölçülen bir backtest
canlıdaki AI'ı temsil etmez → BİRİNCİL hüküm `mc` aşamasından okunur: köprünün kendi çağrısı
(`get_shadow().pwin`, satır başı ~0,19 sn) yıl başına 10 rastgele seansın TÜM satırlarına + ileri dönemin
tamamına uygulanır; kasa kuralının penceresi örneklem günleri boyunca sürer. Üçlü koşul değişmedi. Tüm satırlardaki
eval ölçümü çöküşün kanıtı ve `deep_mlp` kıyası olarak raporlanır. Ölçek sınırı: satırlar her 10. bardan
(`--adim 10`); canlı kasa her barı görür, yani 200 ölçümlük penceresi zaman olarak ~10 kat kısadır.

Aşamalar:  python -m research.noral_282m_backtest --asama indir|veri|tahmin|mc|kiyas|rapor
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SEMBOLLER = (
    "AAPL AMD AMZN AVGO CRWD CSCO CVX DELL GE GOOGL HPQ IBM JPM LMT MRVL MSFT MU NBIS NOW NVDA "
    "ONDS ORCL QCOM QQQ SNDK SPY TSLA TSM V XOM"
).split()
BASLANGIC = "2016-10-06"
BITIS = "2026-10-06"  # hariç: son tam seans 2026-10-05
EGITIM_BAS = pd.Timestamp("2026-07-06")
EGITIM_SON = pd.Timestamp("2026-09-24")  # hariç
KOK = Path.home() / "nihai_hist" / "noral_282m"
MODELLER = {"moe_xl": "neural_moe_xl_live1m_2026-09-23", "deep_mlp": "neural_deep_mlp_live1m_2026-09-23"}
KIPLER = ("canli", "egitim")
# Modelin eğitim satırlarını üreten canlı strateji (Tur-97 öncesi `.env`; yedek ~/nihai_hist/env_yedek_tur97_*).
# Bugünkü canlı strateji `quant_trend` 1m barda aday üretmez → açıkça verilmezse satır kurucu 0 satır döner.
ADAY_STRATEJISI = "event_breakout"
# Gerçek işlem durdurması (indirme hatası değil; yeniden sorgulanarak doğrulandı): NBIS = eski Yandex N.V.,
# 1m işlemler 2022-02-28'de kesilir, 2024-10-21'de döner.
BILINEN_BOSLUK = {"NBIS": (2023,)}
MAX_Z = 6.0  # neural_shadow varsayılanı (NEURAL_SHADOW_MAX_Z); canlı .env ezmiyor
MALIYET_PCT = 0.2  # gidiş-dönüş: %0,1/taraf (projenin varsayılan maliyeti)
MC_GUN_YIL = 10  # canlı yol ~5 satır/sn (ölçüldü): 300 bin satır ~16 saat → yıl başına 10 seanslık gün bloğu
IEX_2Y = Path.home() / "nihai_hist" / "meta_dataset_hist_1m_2y_orta.csv"  # Tur-94 seti (IEX, 520 sembol)
PIT_1H = "meta_dataset_pit_sp500_event_breakout_1h_2016-01-01_alpaca_hist.csv"  # motorun 10 yıllık işlem defteri
SATIR_KOLON = ("symbol", "entry_time", "is_long", "pnl", "pnl_g1", "label")


def _yil_pencereleri() -> list[tuple[str, str]]:
    b, s = pd.Timestamp(BASLANGIC), pd.Timestamp(BITIS)
    out = []
    for y in range(b.year, s.year + 1):
        a, z = max(b, pd.Timestamp(year=y, month=1, day=1)), min(s, pd.Timestamp(year=y + 1, month=1, day=1))
        if a < z:
            out.append((a.strftime("%Y-%m-%d"), z.strftime("%Y-%m-%d")))
    return out


def _yaz(df: pd.DataFrame, hedef: Path) -> None:
    hedef.parent.mkdir(parents=True, exist_ok=True)
    tmp = hedef.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(hedef)


# ───────────────────────── indir ─────────────────────────


def indir(kok: Path, semboller: list[str], bekle_sn: float) -> None:
    """SIP 1m barlarını sembol × yıl parçalarıyla çeker, YALNIZ seans içini saklar (kaldığı yerden devam).

    Yıl parçası: alpaca-py bir çağrının bütün sayfalarını bellekte nesneye çevirir; 10 yıl tek çağrıda
    ~2 milyon bar = GB'larca bellek. Tek iş parçacığı + parça arası bekleme: anahtar canlı hub ile ortak
    (iki indirici 200 istek/dk sınırına dayandı — ölçüldü). Düzeltme kipi `split`: canlı gün-içi barla ve
    Tur-94 setiyle aynı (Tur-91).
    """
    os.environ["ALPACA_HIST_ADJUSTMENT"] = "split"
    from nihai.data import alpaca_history as H
    from nihai.data.data_provider import rth_mask

    for sym in semboller:
        for a, z in _yil_pencereleri():
            f = kok / "sip_rth" / f"{sym}_{a[:4]}.parquet"
            if f.exists():
                continue
            t0 = time.time()
            df = H.get_history(sym, "1m", start=a, end=z, feed="sip", use_cache=False)
            ham = len(df)
            if ham:
                df = df[rth_mask(df.index, 1)]
                df = df[~df.index.duplicated(keep="last")]
            _yaz(df, f)
            sn = time.time() - t0
            print(f"{sym} {a[:4]}: ham {ham} → seans {len(df)} bar, {sn:.1f} sn, ~{ham / 10000 / max(sn, 1e-9) * 60:.0f} sayfa/dk",
                  flush=True)
            time.sleep(bekle_sn)
    print("INDIR BITTI", flush=True)


# ───────────────────────── veri ─────────────────────────


def _yil_sayimi(kok: Path, sym: str) -> list[int] | None:
    ps = [kok / "sip_rth" / f"{sym}_{a[:4]}.parquet" for a, _ in _yil_pencereleri()]
    if not all(p.exists() for p in ps):
        return None
    return [len(pd.read_parquet(p, columns=[])) for p in ps]


def _sembol_barlari(kok: Path, sym: str) -> pd.DataFrame:
    parcalar = [pd.read_parquet(kok / "sip_rth" / f"{sym}_{a[:4]}.parquet") for a, _ in _yil_pencereleri()]
    parcalar = [d for d in parcalar if len(d)]
    if not parcalar:
        return pd.DataFrame()
    df = pd.concat(parcalar).sort_index()
    return df[~df.index.duplicated(keep="last")]


def _veri_isci(kok: str, sym: str, adim: int) -> tuple[str, int, float, str]:
    """Sembolün 10 yıllık aday satırları + aynı kararın 1 bar GEÇ uygulanmış getirisi (`pnl_g1`)."""
    import research.build_hist_1m_dataset as B
    from nihai.execution.prediction_tracker import _canon_ts, _pos_map

    t0 = time.time()
    k = Path(kok)
    sayim = _yil_sayimi(k, sym) or []
    dolu = [i for i, n in enumerate(sayim) if n]
    yillar = [int(a[:4]) for a, _ in _yil_pencereleri()]
    # indirici hatada boş çerçeve döndürür: ilk dolu yıldan SONRA boş yıl = veri deliği, sessizce geçilmez
    if dolu and any(n == 0 and y not in BILINEN_BOSLUK.get(sym, ()) for n, y in zip(sayim[dolu[0]:], yillar[dolu[0]:], strict=True)):
        return sym, -1, 0.0, f"EKSİK YIL (indirme hatası?): {sayim}"
    df = _sembol_barlari(k, sym)
    rows = B.sembol_satirlari(sym, df, adim=adim, strateji=ADAY_STRATEJISI) if len(df) else []
    out = pd.DataFrame(rows, columns=[*B.FEATURE_NAMES, "pnl", "label", "symbol", "entry_time"])
    g1: list[float] = []
    if len(out):
        pm = _pos_map(df)
        for z, u in zip(out["entry_time"], out["is_long"], strict=True):
            i = pm.get(_canon_ts(z))
            r = None if i is None else B.taraf_getirisi(df, pm, i + 1, "LONG" if u > 0.5 else "SHORT")
            g1.append(np.nan if r is None else r)
    out["pnl_g1"] = np.asarray(g1, dtype=float)
    _yaz(out, k / "satir" / f"{sym}.parquet")
    return sym, len(out), time.time() - t0, ""


def veri(kok: Path, semboller: list[str], adim: int, isci: int, izle: bool) -> None:
    """İndirmesi TAMAMLANAN sembolleri satıra çevirir; `izle` ile indirme bitene kadar bekler."""
    bekleyen = [s for s in semboller if not (kok / "satir" / f"{s}.parquet").exists()]
    with ProcessPoolExecutor(max_workers=max(1, isci)) as ex:
        isler: dict = {}
        while True:
            for s in [s for s in bekleyen if _yil_sayimi(kok, s) is not None]:
                isler[ex.submit(_veri_isci, str(kok), s, adim)] = s
                bekleyen.remove(s)
            for f in [f for f in isler if f.done()]:
                sym, n, sn, not_ = f.result()
                print(f"{sym}: {n} satır, {sn:.0f} sn {not_}", flush=True)
                isler.pop(f)
            if not isler and (not bekleyen or not izle):
                break
            time.sleep(5)
    print(f"VERI BITTI (indirmesi eksik: {bekleyen})", flush=True)


# ───────────────────────── tahmin ─────────────────────────


def _yukle(ad: str):
    from nihai.ai import neural as N
    from nihai.config import DATA_DIR

    m = N.load_any(DATA_DIR / MODELLER[ad])
    if m is None:
        raise SystemExit(f"model yüklenemedi: {MODELLER[ad]}")
    return m


def _modeller(aygit: str) -> dict:
    from nihai.ai import neural as N

    N._INFER_BATCH = 1024  # yalnız bu süreç: GPU canlı süreçle paylaşılıyor, VRAM zirvesi küçük kalsın
    out = {}
    for ad in MODELLER:
        out[ad] = _yukle(ad)
        if ad == "moe_xl":
            out[ad].to_device(aygit)
    return out


def notr_kolonlar(m) -> list[int]:
    """Eğitimde sabit olduğu için nötrlenen kolonlar: manifestte μ=0 ve σ=1 izi bırakır."""
    return [j for j, (mu, sg) in enumerate(zip(m._feat_mu, m._feat_sigma, strict=True)) if mu == 0.0 and sg == 1.0]


def tahmin_kolonlari(df: pd.DataFrame, modeller: dict) -> pd.DataFrame:
    """Her model için canlı OOD ölçüsü (max|z|, ham vektörde) + iki besleme kipinde p (eval, dropout kapalı)."""
    out = df[[c for c in SATIR_KOLON if c in df.columns]].copy()
    for ad, m in modeller.items():
        X = df[list(m.feature_names)].to_numpy(np.float32)
        Xe = X.copy()
        Xe[:, notr_kolonlar(m)] = 0.0
        out[f"z_{ad}"] = m.ood_zmax(X)
        out[f"p_{ad}_canli"] = m.predict_proba(X)
        out[f"p_{ad}_egitim"] = m.predict_proba(Xe)
    return out


def _iex2y(semboller: list[str]) -> pd.DataFrame:
    if not IEX_2Y.exists():
        return pd.DataFrame()
    s = set(semboller)
    return pd.concat([c[c["symbol"].isin(s)] for c in pd.read_csv(IEX_2Y, chunksize=250_000)], ignore_index=True)


def _pit1h() -> pd.DataFrame:
    from nihai.config import DATA_DIR

    p = DATA_DIR / PIT_1H
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def tahmin(kok: Path, semboller: list[str], aygit: str) -> None:
    isler: list[tuple[Path, Callable[[], pd.DataFrame]]] = [
        (kok / "tahmin" / f"{s}.parquet", lambda s=s: pd.read_parquet(kok / "satir" / f"{s}.parquet"))  # type: ignore[misc]
        for s in semboller if (kok / "satir" / f"{s}.parquet").exists()]
    isler += [(kok / "tahmin" / "_iex2y.parquet", lambda: _iex2y(semboller)), (kok / "tahmin" / "_pit1h.parquet", _pit1h)]
    bekleyen = [(h, f) for h, f in isler if not h.exists()]
    if not bekleyen:
        print("TAHMIN: hepsi hazır", flush=True)
        return
    mods = _modeller(aygit)
    for h, f in bekleyen:
        t0 = time.time()
        df = f()
        if not len(df):
            continue
        _yaz(tahmin_kolonlari(df, mods), h)
        sn = time.time() - t0
        print(f"{h.stem}: {len(df)} satır, {sn:.0f} sn ({len(df) / max(sn, 1e-9):.0f} satır/sn)", flush=True)
    print("TAHMIN BITTI", flush=True)


# ───────────────────────── mc: canlı yolun kendisi ─────────────────────────


def mc_gunleri(kok: Path) -> list[pd.Timestamp]:
    """Canlı yol örneklemi: eğitim öncesi her takvim yılından MC_GUN_YIL rastgele seans + eğitim döneminden 3
    + İLERİ dönemin tamamı. Gün BLOĞU: kasa kuralı aynı günün önceki ölçümlerine bakar, seyrek satır örneği
    o pencereyi haftalara yayardı."""
    idx = [pd.read_parquet(kok / "sip_rth" / f"AAPL_{a[:4]}.parquet", columns=[]).index for a, _ in _yil_pencereleri()]
    gunler = pd.DatetimeIndex(sorted({g for i in idx for g in i.normalize().unique()}))
    rng = np.random.default_rng(107)
    once = gunler[gunler < EGITIM_BAS]
    sec: list = []
    for y in sorted(set(once.year)):
        g = once[once.year == y]
        sec += list(rng.choice(g, size=min(MC_GUN_YIL, len(g)), replace=False))
    eg = gunler[(gunler >= EGITIM_BAS) & (gunler < EGITIM_SON)]
    sec += list(rng.choice(eg, size=min(3, len(eg)), replace=False)) + list(gunler[gunler >= EGITIM_SON])
    return sorted(pd.Timestamp(g) for g in sec)


def mc(kok: Path, semboller: list[str], izle: bool) -> None:
    """Örneklem günlerinin satırlarına CANLI köprünün kendi çağrısıyla p üretir: `neural_shadow.get_shadow().pwin`
    (kart seçimi, OOD kapısı, MC-dropout T=20, aygıt — hepsi canlı süreçteki gibi). Dropout kapalı `predict_proba`
    bu modelde sabit bir sayı verir (kıyas: yayılım ~1e-7); canlının yazdığı değer MC ortalamasıdır."""
    from nihai.execution.neural_shadow import get_shadow
    from nihai.signals.meta_features import FEATURE_NAMES

    gunler = set(mc_gunleri(kok))
    sh = get_shadow()
    bekleyen = list(semboller)
    while True:
        hazir = [s for s in bekleyen if (kok / "satir" / f"{s}.parquet").exists()]
        for s in hazir:
            bekleyen.remove(s)
            h = kok / "mc" / f"{s}.parquet"
            if h.exists():
                continue
            df = pd.read_parquet(kok / "satir" / f"{s}.parquet")
            o = df[pd.to_datetime(df["entry_time"]).dt.normalize().isin(gunler)] if len(df) else df
            t0 = time.time()
            r = [sh.pwin(dict(zip(FEATURE_NAMES, map(float, x), strict=True)), "LONG" if u > 0.5 else "SHORT")
                 for x, u in zip(o[FEATURE_NAMES].to_numpy(), o["is_long"], strict=True)]
            st = sh.status()
            if len(o) and st.get("tier") != "quant_transformer_moe_xl":
                raise SystemExit(f"canlı köprü 282M'i seçmedi: {st}")
            out = o[[c for c in SATIR_KOLON if c in o.columns]].copy()
            out["p_mc"] = [np.nan if x is None else x[0] for x in r]
            out["mc_std"] = [np.nan if x is None else x[1] for x in r]
            _yaz(out, h)
            print(f"{s}: {len(o)} satır, {time.time() - t0:.0f} sn · köprü n_ok={st.get('n_ok')} n_ood={st.get('n_ood')} "
                  f"aygıt={st.get('device')} max_z={st.get('max_z')}", flush=True)
        if not bekleyen or not izle:
            break
        if not hazir:
            time.sleep(10)
    print(f"MC BITTI (satırı olmayan: {bekleyen})", flush=True)


# ───────────────────────── kıyas (hız/bellek) ─────────────────────────


def _medyan_sn(f, tekrar: int) -> float:
    f()  # ısınma
    ts = []
    for _ in range(tekrar):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def govde_katkisi(m, X: np.ndarray) -> dict | None:
    """Blokların (dikkat + uzmanlar) çıktıya katkısı: LayerScale γ'ları GEÇİCİ olarak 0 yapılır (artık dallar
    kapalı), çıktı farkı ölçülür, γ'lar geri yazılır. Girdi [CLS]'e yalnız bu dallardan ulaşır; γ başlangıç
    değerinde kalmışsa gövde kapalıdır ve model girdiye bakamaz. LayerScale'siz basamakta None."""
    import torch

    from nihai.ai import neural as N

    adli = list(m._model.named_parameters())
    gam = [p for ad, p in adli if ad.endswith(".gamma")]
    if not gam:
        return None
    p0 = m.predict_proba(X)
    yedek = [g.detach().clone() for g in gam]
    try:
        with torch.no_grad():
            for g in gam:
                g.zero_()
        p1 = m.predict_proba(X)
    finally:
        with torch.no_grad():
            for g, y in zip(gam, yedek, strict=True):
                g.copy_(y)
    blok = sum(p.numel() for ad, p in adli if ad.startswith("blocks."))
    return {"gamma_ort_mutlak": float(torch.cat([y.abs().flatten() for y in yedek]).mean()),
            "gamma_baslangic": getattr(N, "_XL_LAYERSCALE_INIT", None), "blok_parametre_payi": blok / m.param_count(),
            "govdesiz_maks_fark": float(np.abs(p1 - p0).max()), "p_std": float(p0.std()), "govdesiz_p_std": float(p1.std())}


def _canli_uyum(m, X: np.ndarray, buyuk: bool, n: int = 300) -> dict:
    """Canlı değer (tek aday, MC-dropout T=20) ↔ dropout kapalı tek geçiş; ve iki besleme kipi (ham ↔ eğitimdeki
    gibi nötr kolonlar 0). Canlı eğitim setinin eşit aralıklı `n` satırında."""
    ix = np.linspace(0, len(X) - 1, n).astype(int)
    Xe = X.copy()
    Xe[:, notr_kolonlar(m)] = 0.0

    def mc_p(A: np.ndarray) -> np.ndarray:
        return np.array([m.predict_with_uncertainty(A[i:i + 1], T=20, batched=buyuk) for i in ix])[:, :, 0]

    mc, mce, ev, eve = mc_p(X), mc_p(Xe), m.predict_proba(X[ix]), m.predict_proba(Xe[ix])
    # girdi duyarlılığı: eğitim ortalaması (z=0) ↔ TÜM özellikler ±3σ; öğrenmiş bir model bu üçüne aynı p'yi vermez
    mu, sg = np.asarray(m._feat_mu, np.float32), np.asarray(m._feat_sigma, np.float32)
    uc = m.predict_proba(np.stack([mu, mu + 3 * sg, mu - 3 * sg]))

    def kor(a: np.ndarray, b: np.ndarray, sira: bool = False) -> float | None:
        k = pd.Series(a).corr(pd.Series(b), method="spearman" if sira else "pearson")
        return None if not np.isfinite(k) else float(k)

    return {"mc_eval_uyumu": {"n": n, "pearson": kor(mc[:, 0], ev), "spearman": kor(mc[:, 0], ev, sira=True),
                              "ort_mutlak_fark": float(np.abs(mc[:, 0] - ev).mean()), "p_std_mc": float(mc[:, 0].std()),
                              "p_std_eval": float(ev.std()), "mc_ic_std_ort": float(mc[:, 1].mean()),
                              "p_ort_mc": float(mc[:, 0].mean()), "p_ort_eval": float(ev.mean())},
            "girdi_duyarliligi": {"p_z0": float(uc[0]), "p_z+3": float(uc[1]), "p_z-3": float(uc[2])},
            "govde_katkisi": govde_katkisi(m, X[ix]),
            "besleme_kaymasi": {"notr_kolonlar": [m.feature_names[j] for j in notr_kolonlar(m)],
                                "mc_ort_canli": float(mc[:, 0].mean()), "mc_ort_egitim": float(mce[:, 0].mean()),
                                "mc_pearson": kor(mc[:, 0], mce[:, 0]), "mc_std_egitim": float(mce[:, 0].std()),
                                "eval_ort_canli": float(ev.mean()), "eval_ort_egitim": float(eve.mean()),
                                "eval_std_egitim": float(eve.std()), "eval_pearson": kor(ev, eve)}}


def _kiyas_model(ad: str, canli: pd.DataFrame, cuda: bool) -> dict:
    import torch

    from nihai.ai import neural as N
    from nihai.config import DATA_DIR

    t = time.perf_counter()
    m = _yukle(ad)
    r: dict = {"tier": m.tier, "parametre": int(m.param_count()), "yukleme_sn": time.perf_counter() - t,
               "agirlik_mb": (DATA_DIR / f"{MODELLER[ad]}.pt").stat().st_size / 2**20}
    X = canli[list(m.feature_names)].to_numpy(np.float32)
    buyuk = ad == "moe_xl"

    def tek() -> np.ndarray:
        return m.predict_proba(X[:1])

    def mc20() -> tuple:  # canlı köprünün çağrısı: tek aday, T=20, büyük basamakta toplu
        return m.predict_with_uncertainty(X[:1], T=20, batched=buyuk)

    with N.inference_threads(1):  # canlı köprü tek iş parçacığıyla çağırır (Tur-80)
        r["cpu_tek_satir_ms"] = 1e3 * _medyan_sn(tek, 3 if buyuk else 50)
        r["cpu_mc20_ms"] = 1e3 * _medyan_sn(mc20, 2 if buyuk else 30)
        if not buyuk:  # deep_mlp canlıda CPU + döngü ile koşar
            r.update(_canli_uyum(m, X, buyuk))
    if not cuda:
        return r
    t = time.perf_counter()
    m.to_device("cuda")
    r["gpu_tasima_sn"] = time.perf_counter() - t
    model_mb = torch.cuda.memory_allocated() / 2**20
    r["gpu_model_mb"] = model_mb
    r["gpu_tek_satir_ms"] = 1e3 * _medyan_sn(tek, 30)
    r["gpu_mc20_ms"] = 1e3 * _medyan_sn(mc20, 20)
    r["gpu_toplu"] = []
    onceki = None
    for b in (1, 16, 128, 1024, 4096):
        bos_mb = torch.cuda.mem_get_info()[0] / 2**20
        if onceki and (onceki[1] - model_mb) * b / onceki[0] > 0.8 * bos_mb:
            r["gpu_toplu"].append({"toplu": b, "atlandi": f"tahmini zirve boş VRAM'i ({bos_mb:.0f} MB) aşıyor"})
            break
        torch.cuda.reset_peak_memory_stats()
        sn = _medyan_sn(lambda b=b: m.predict_proba(X[:b]), 5 if b >= 1024 else 15)
        zirve = torch.cuda.max_memory_allocated() / 2**20
        r["gpu_toplu"].append({"toplu": b, "ms": 1e3 * sn, "satir_sn": b / sn, "zirve_vram_mb": zirve})
        onceki = (b, zirve)
    if buyuk:  # 282M canlıda GPU + toplu MC ile koşar
        r.update(_canli_uyum(m, X, buyuk))
    m.to_device("cpu")
    torch.cuda.empty_cache()
    return r


def kiyas(aygit: str) -> dict:
    """Gecikme · iş hacmi · bellek. Canlı yolun kendi çağrılarıyla (eval tek geçiş + MC-dropout T=20)."""
    import torch

    from nihai.config import DATA_DIR

    canli = pd.read_csv(DATA_DIR / "meta_dataset_live_1m_2026-09-23b.csv")
    cuda = aygit != "cpu" and torch.cuda.is_available()
    return {"gpu": torch.cuda.get_device_name(0) if cuda else None, "torch": torch.__version__,
            "modeller": {ad: _kiyas_model(ad, canli, cuda) for ad in MODELLER}}


# ───────────────────────── rapor ─────────────────────────


def hazirla(df: pd.DataFrame) -> pd.DataFrame:
    """Kronolojik sıra, gün kümesi, NEDENSEL sabit taban (o günden önceki tüm satırların kazanma oranı), dönem."""
    df = df.copy()
    df["zaman"] = pd.to_datetime(df["entry_time"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.sort_values(["zaman", "symbol"], kind="stable").reset_index(drop=True)
    df["gun"] = df["zaman"].dt.normalize()
    g = df.groupby("gun")["label"].agg(["sum", "count"]).sort_index()
    onceki = (g["sum"].cumsum() - g["sum"]) / (g["count"].cumsum() - g["count"]).replace(0, np.nan)
    df["taban"] = df["gun"].map(onceki).fillna(0.5)
    df["donem"] = np.where(df["zaman"] < EGITIM_BAS, "oncesi", np.where(df["zaman"] < EGITIM_SON, "egitim", "ileri"))
    return df


def _gun_no(d: pd.DataFrame) -> np.ndarray:
    return d["gun"].to_numpy().astype("datetime64[D]").astype(np.int64)


def olc(d: pd.DataFrame, pk: str) -> dict:
    """AUC + Brier ↔ nedensel sabit taban (eşleştirilmiş, gün-kümeli t; NEGATİF = model tabandan iyi)."""
    from research.horizon_sweep import _cluster_t
    from research.train_meta import _auc

    if len(d) < 200 or d["label"].nunique() < 2:
        return {"n": len(d), "hukum": "YETERSIZ"}
    y, p, b = (d[k].to_numpy(float) for k in ("label", pk, "taban"))
    t, kume = _cluster_t((p - y) ** 2 - (b - y) ** 2, _gun_no(d))
    ic = [(_auc(s["label"].to_numpy(), s[pk].to_numpy()), len(s)) for _, s in d.groupby("symbol")
          if len(s) >= 200 and s["label"].nunique() == 2]
    bm, bt = float(((p - y) ** 2).mean()), float(((b - y) ** 2).mean())
    return {"n": len(d), "gun": int(kume), "taban_orani": float(y.mean()), "p_ort": float(p.mean()),
            "p_std": float(p.std()), "p_min": float(p.min()), "p_max": float(p.max()), "auc": float(_auc(y, p)),
            "auc_sembol_ici": float(np.average([a for a, _ in ic], weights=[n for _, n in ic])) if ic else None,
            "brier": bm, "brier_taban": bt, "bss": 1.0 - bm / bt, "t_gun": None if t is None else float(t)}


def kasa_kurali(p: pd.Series) -> np.ndarray:
    """Canlı gölge kasanın `noral` kuralı (golge_kasalar): p ≥ modelin KENDİ önceki 200 ölçümünün medyanı;
    30 ölçümden önce girmez. Dönüş: 1 tut · 0 ele · −1 ısınma. Aday kendi eşiğine girmez (shift); pencere
    seanslar arası taşınır (canlı kuyruk da öyle)."""
    from nihai.execution.golge_kasalar import NORAL_ISINMA, NORAL_PENCERE

    e = p.shift(1).rolling(NORAL_PENCERE, min_periods=NORAL_ISINMA).median()
    return np.where(e.isna().to_numpy(), -1, (p.to_numpy() >= e.to_numpy()).astype(int))


def kapi_ozet(d: pd.DataFrame, tut: np.ndarray, kol: str, rng: np.random.Generator, carpan: float = 100.0) -> dict:
    """Kapı simülasyonu: tutulan ↔ elenen ↔ taban (carpan=100 → % getiriden bps), gün-kümeli t, rastgele kapı."""
    from research.shadow_promote import _cluster_t

    r = d[kol].to_numpy(float)
    m = (tut >= 0) & np.isfinite(r)
    r, k, g = r[m] * carpan, tut[m] == 1, _gun_no(d)[m]
    if k.sum() < 30 or (~k).sum() < 30:
        return {"n": int(m.sum()), "tutulan": int(k.sum()), "hukum": "YETERSIZ"}
    t, pv = _cluster_t(r, k, list(g))
    rast = np.array([r[rng.permutation(k)].mean() for _ in range(200)])
    return {"n": int(m.sum()), "tutulan_pay": float(k.mean()), "taban": float(r.mean()), "tutulan": float(r[k].mean()),
            "elenen": float(r[~k].mean()), "fazlalik": float(r[k].mean() - r.mean()), "t_gun": float(t), "p": float(pv),
            "rastgele_yuzdelik": float((rast < r[k].mean()).mean() * 100),
            "isabet_tutulan": float((r[k] > 0).mean()), "isabet_taban": float((r > 0).mean())}


def seri_ozet(x_pct: np.ndarray) -> dict:
    """Günlük % getiri serisi → bileşik toplam, yıllık, Sharpe, azami düşüş."""
    x = np.asarray(x_pct, float) / 100.0
    eq = np.cumprod(1.0 + x)
    sd = x.std(ddof=1)
    return {"gun": len(x), "toplam_pct": float((eq[-1] - 1) * 100), "yillik_pct": float((eq[-1] ** (252.0 / len(x)) - 1) * 100),
            "sharpe": float(x.mean() / sd * np.sqrt(252)) if sd > 0 else None,
            "mdd_pct": float((eq / np.maximum.accumulate(eq) - 1).min() * 100)}


def gunluk_seriler(d: pd.DataFrame, tut: np.ndarray, spy: pd.Series, surekli: bool = False) -> dict:
    """Kaldıraçsız günlük getiri: sermaye o günün işlemlerine eşit bölünür (gün = işlemlerin ortalaması).
    `surekli`: ilk ve son aday günü arasındaki TÜM seanslar (adaysız gün = 0); değilse yalnız aday günleri."""
    gunler = pd.DatetimeIndex(sorted(d["gun"].unique()))
    if surekli:
        gunler = pd.DatetimeIndex(spy.index[(spy.index >= gunler[0]) & (spy.index <= gunler[-1])])
    out = {}
    for ad, maske in (("taban_tum_adaylar", tut >= 0), ("kapi", tut == 1)):
        for etiket, maliyet in (("brut", 0.0), ("net", MALIYET_PCT)):
            s = (d["pnl"][maske] - maliyet).groupby(d["gun"][maske]).mean().reindex(gunler).fillna(0.0)
            out[f"{ad}_{etiket}"] = seri_ozet(s.to_numpy())
    out["spy_al_tut"] = seri_ozet(spy.reindex(gunler).fillna(0.0).to_numpy())
    return out


def dilim_karnesi(df: pd.DataFrame, ic: np.ndarray, pk: str, rng: np.random.Generator, carpan: float,
                  dilimler: dict) -> dict:
    """Bir p kolonunun karnesi: dilim başına dağılım-içi pay + olc + kapı (+1 bar gecikme varsa o da).
    Kapı eşiği dilimden bağımsız, dağılım-içi satırların TÜM kronolojik serisinde kurulur."""
    d = df[ic].reset_index(drop=True)
    tut = kasa_kurali(d[pk])
    k: dict = {}
    for dilim, maske_f in dilimler.items():
        mt, md = maske_f(df).to_numpy(), maske_f(d).to_numpy()
        if not mt.any():
            continue
        s = {"n_tum": int(mt.sum()), "dagilim_ici": float(ic[mt].mean()), **olc(d[md], pk),
             "kapi": kapi_ozet(d[md], tut[md], "pnl", rng, carpan)}
        if "pnl_g1" in d.columns:
            s["kapi_g1"] = kapi_ozet(d[md], tut[md], "pnl_g1", rng, carpan)
        k[dilim] = s
    return k


def _karne(df: pd.DataFrame, rng: np.random.Generator, carpan: float, dilimler: dict) -> dict:
    """Eval kipi (dropout kapalı) karnesi: her model × besleme kipi."""
    return {f"{ad}_{kip}": dilim_karnesi(df, (df[f"z_{ad}"] <= MAX_Z).to_numpy(), f"p_{ad}_{kip}", rng, carpan, dilimler)
            for ad in MODELLER for kip in KIPLER}


def _gunluk(df: pd.DataFrame, ic: np.ndarray, pk: str, spy: pd.Series, surekli: bool) -> dict:
    """Eğitim öncesi dönemde kapı ↔ taban ↔ SPY günlük serileri."""
    d = df[ic].reset_index(drop=True)
    tut = kasa_kurali(d[pk])
    m = (d["donem"] == "oncesi").to_numpy()
    return gunluk_seriler(d[m].reset_index(drop=True), tut[m], spy, surekli)


def _ozellik_kaymasi(kok: Path, ortak: list[str], bas: pd.Timestamp) -> dict:
    """SIP ↔ IEX, aynı semboller ve dönem: özellik ortalamaları farkı (IEX std'si biriminde), en büyük altısı."""
    from nihai.signals.meta_features import FEATURE_NAMES

    sip = pd.concat([pd.read_parquet(kok / "satir" / f"{s}.parquet") for s in ortak], ignore_index=True)
    iex = _iex2y(ortak)
    kesit = []
    for d in (sip, iex):
        z = pd.to_datetime(d["entry_time"], format="mixed", utc=True).dt.tz_localize(None)
        kesit.append(d[(z >= bas) & (z < EGITIM_BAS)])
    sip, iex = kesit
    fark = ((sip[FEATURE_NAMES].mean() - iex[FEATURE_NAMES].mean()) / iex[FEATURE_NAMES].std().replace(0, np.nan)).dropna()
    sirali = fark.reindex(fark.abs().sort_values(ascending=False).index)
    return {"n_sip": len(sip), "n_iex": len(iex), "en_buyuk": sirali.head(6).round(3).to_dict()}


def _hukum(s: dict) -> dict:
    """Ön-kayıtlı üçlü koşul (eğitim öncesi dönem)."""
    k, g = s.get("kapi", {}), s.get("kapi_g1", {})
    kosul = {"brier_t<-1.645": (s.get("t_gun") is not None and s["t_gun"] < -1.645),
             "kapi_t>1.645": k.get("t_gun", 0.0) > 1.645, "gecikmede_isaret": g.get("fazlalik", -1.0) > 0}
    return {"kosullar": kosul, "hukum": "BECERI_VAR" if all(kosul.values()) else "BECERI_YOK",
            "cikti_cokmus": bool(s.get("p_std", 1.0) < 1e-3)}


def rapor(kok: Path, semboller: list[str]) -> dict:
    from nihai.config import DATA_DIR

    rng = np.random.default_rng(107)
    ps = [kok / "tahmin" / f"{s}.parquet" for s in semboller]
    sip = hazirla(pd.concat([pd.read_parquet(p) for p in ps if p.exists()], ignore_index=True))
    donem = {dn: (lambda x, dn=dn: x["donem"] == dn) for dn in ("oncesi", "egitim", "ileri")}
    yillar = {str(y): (lambda x, y=y: (x["donem"] == "oncesi") & (x["zaman"].dt.year == y))
              for y in sorted(sip["zaman"].dt.year.unique())}
    out: dict = {"uretim": time.strftime("%Y-%m-%d %H:%M"), "pencere": [BASLANGIC, BITIS], "max_z": MAX_Z,
                 "maliyet_pct": MALIYET_PCT, "satir": len(sip), "sembol": int(sip["symbol"].nunique()),
                 "sembol_ilk_satir": {s: str(g.min().date()) for s, g in sip.groupby("symbol")["zaman"]},
                 "sip10y": _karne(sip, rng, 100.0, {**donem, **yillar})}
    out["hukum"] = {k: _hukum(v["oncesi"]) for k, v in out["sip10y"].items() if "oncesi" in v}
    spy = _sembol_barlari(kok, "SPY")["close"]
    spy = spy.groupby(spy.index.normalize()).last().pct_change().dropna() * 100
    out["gunluk"] = {f"{ad}_eval_tum_donem": _gunluk(sip, (sip[f"z_{ad}"] <= MAX_Z).to_numpy(), f"p_{ad}_canli", spy, True)
                     for ad in MODELLER}
    # CANLI YOL (MC-dropout, köprünün kendi çağrısı): örneklem günleri; taban tüm satırlardan (nedensel)
    ps = [kok / "mc" / f"{s}.parquet" for s in semboller]
    if any(p.exists() for p in ps):
        mcd = hazirla(pd.concat([pd.read_parquet(p) for p in ps if p.exists()], ignore_index=True))
        mcd["taban"] = mcd["gun"].map(sip.groupby("gun")["taban"].first()).fillna(mcd["taban"])
        ic = mcd["p_mc"].notna().to_numpy()
        pm = mcd["p_mc"][ic]
        out["canli_yol"] = {"satir": len(mcd), "gun": int(mcd["gun"].nunique()), "mc_ic_std_ort": float(mcd["mc_std"].mean()),
                            "p_tekil_deger": int(pm.nunique()), "p_en_sik_deger": float(pm.mode().iloc[0]),
                            "p_en_sik_pay": float((pm == pm.mode().iloc[0]).mean()),
                            "p_yuzdelik": {str(q): float(pm.quantile(q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
                            **dilim_karnesi(mcd, ic, "p_mc", rng, 100.0, {**donem, **yillar})}
        out["hukum"]["canli_yol"] = _hukum(out["canli_yol"].get("oncesi", {}))
        out["gunluk"]["canli_yol_ornek_gunler"] = _gunluk(mcd, ic, "p_mc", spy, False)
    # besleme farkı: aynı semboller, iki beslemenin de bulunduğu eğitim-öncesi pencere
    f = kok / "tahmin" / "_iex2y.parquet"
    if f.exists():
        iex = hazirla(pd.read_parquet(f))
        ortak = sorted(set(iex["symbol"]))
        bas = iex["zaman"].min().normalize()
        pencere = {"ortak": lambda x: (x["zaman"] >= bas) & (x["donem"] == "oncesi")}
        out["besleme"] = {"sembol": len(ortak), "baslangic": str(bas.date()),
                          "sip": _karne(sip[sip["symbol"].isin(ortak)].reset_index(drop=True), rng, 100.0, pencere),
                          "iex": _karne(iex, rng, 100.0, pencere),
                          "ozellik_kaymasi": _ozellik_kaymasi(kok, ortak, bas)}
    f = kok / "tahmin" / "_pit1h.parquet"
    if f.exists():  # motorun gerçek işlem defteri (1h bar, PIT S&P500, pnl $): model 1m'de eğitildi → OOD beklenir
        pit = hazirla(pd.read_parquet(f))
        pit_yil = {str(y): (lambda x, y=y: x["zaman"].dt.year == y) for y in sorted(pit["zaman"].dt.year.unique())}
        out["pit1h"] = {"satir": len(pit), "aralik": [str(pit["zaman"].min().date()), str(pit["zaman"].max().date())],
                        **_karne(pit, rng, 1.0, {"oncesi": donem["oncesi"], **pit_yil})}
    (DATA_DIR / "noral_282m_karne.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    return out


def _json(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def _f(v, bicim: str = ".4f") -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, bicim)


def _tablo(baslik: str, k: dict, birim: str = "bps") -> None:
    print(f"\n## {baslik}")
    print(f"dilim   |      n | dağ-içi | taban  | p_ort   | p_std     | AUC    | AUC-smb | BSS      | t_gün  | tut%  | "
          f"fazlalık {birim} | t_gün | rast% | +1bar")
    for dilim, s in k.items():
        if not isinstance(s, dict) or "n_tum" not in s:
            continue
        if s.get("hukum") == "YETERSIZ":
            print(f"{dilim:7s} | {s['n_tum']:6d} | {_f(s['dagilim_ici'], '.3f')}   | yetersiz örneklem")
            continue
        kp, g1 = s["kapi"], s.get("kapi_g1", {})
        print(f"{dilim:7s} | {s['n_tum']:6d} | {_f(s['dagilim_ici'], '.3f')}   | {_f(s['taban_orani'])} | {_f(s['p_ort'], '.5f')} | "
              f"{_f(s['p_std'], '.2e')}  | {_f(s['auc'])} | {_f(s['auc_sembol_ici'])}  | {_f(s['bss'], '+.5f')} | "
              f"{_f(s['t_gun'], '+.2f')}  | {_f(kp.get('tutulan_pay'), '.3f')} | {_f(kp.get('fazlalik'), '+.3f')}       | "
              f"{_f(kp.get('t_gun'), '+.2f')} | {_f(kp.get('rastgele_yuzdelik'), '.0f')}   | {_f(g1.get('fazlalik'), '+.3f')}")


def yazdir(k: dict) -> None:
    print(f"\n# 282M 10 YILLIK BACKTEST — {k['satir']:,} aday satır · {k['sembol']} sembol · {k['pencere'][0]} → {k['pencere'][1]}")
    if "canli_yol" in k:
        c = k["canli_yol"]
        _tablo(f"282M · CANLI YOL (MC-dropout T=20, köprünün kendi çağrısı) — {c['satir']:,} satır / {c['gun']} örneklem günü", c)
    for anahtar, baslik in (("moe_xl_canli", "282M · eval (dropout kapalı) · canlı besleme"),
                            ("moe_xl_egitim", "282M · eval · eğitim beslemesi"),
                            ("deep_mlp_canli", "deep_mlp 186k · eval · canlı besleme"),
                            ("deep_mlp_egitim", "deep_mlp 186k · eval · eğitim beslemesi")):
        _tablo(baslik, k["sip10y"][anahtar])
    print("\n## Ön-kayıtlı hüküm (eğitim öncesi dönem)")
    for a, h in k["hukum"].items():
        print(f"{a}: {h['hukum']} {h['kosullar']} çıktı çökmüş={h['cikti_cokmus']}")
    print("\n## Günlük seri (eğitim öncesi; sermaye güne eşit bölünür, kaldıraçsız; net = işlem başı %0,2) ↔ SPY fiyat getirisi")
    for ad, g in k["gunluk"].items():
        for a, s in g.items():
            print(f"{ad:26s} {a:24s} {s['gun']:5d} gün · toplam {_f(s['toplam_pct'], '+.2f')}% · yıllık {_f(s['yillik_pct'], '+.2f')}% · "
                  f"Sharpe {_f(s['sharpe'], '+.2f')} · MDD {_f(s['mdd_pct'], '.2f')}%")
    if "besleme" in k:
        b = k["besleme"]
        print(f"\n## Besleme farkı: {b['sembol']} ortak sembol, {b['baslangic']} → eğitim öncesi · özellik kayması {b['ozellik_kaymasi']}")
        for f in ("sip", "iex"):
            for a in ("moe_xl_canli", "deep_mlp_canli"):
                _tablo(f"{f.upper()} · {a}", b[f][a])
    if "pit1h" in k:
        p = k["pit1h"]
        print(f"\n## Motorun 10 yıllık işlem defteri (1h bar, PIT S&P 500): {p['satir']:,} işlem · {p['aralik'][0]} → {p['aralik'][1]}")
        for a in ("moe_xl_canli", "deep_mlp_canli"):
            _tablo(a, p[a], birim="$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--asama", required=True, choices=["indir", "veri", "tahmin", "mc", "kiyas", "rapor"])
    ap.add_argument("--kok", default=str(KOK), help="ara dosya kökü (OneDrive DIŞI)")
    ap.add_argument("--semboller", default=",".join(SEMBOLLER))
    ap.add_argument("--bekle-sn", type=float, default=1.0, help="indirme parçaları arası bekleme")
    ap.add_argument("--adim", type=int, default=5, help="her kaçıncı bar değerlendirilir (5 = ufuk, Tur-94 ile aynı)")
    ap.add_argument("--isci", type=int, default=3)
    ap.add_argument("--izle", action="store_true", help="veri: indirme bitene kadar yeni sembolleri bekle")
    ap.add_argument("--aygit", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    kok, semboller = Path(a.kok), a.semboller.split(",")
    if a.asama == "indir":
        indir(kok, semboller, a.bekle_sn)
    elif a.asama == "veri":
        veri(kok, semboller, a.adim, a.isci, a.izle)
    elif a.asama == "tahmin":
        tahmin(kok, semboller, a.aygit)
    elif a.asama == "mc":
        mc(kok, semboller, a.izle)
    elif a.asama == "kiyas":
        from nihai.config import DATA_DIR

        k = kiyas(a.aygit)
        (DATA_DIR / "noral_282m_kiyas.json").write_text(json.dumps(k, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
        print(json.dumps(k, ensure_ascii=False, indent=1, default=_json))
    else:
        yazdir(rapor(kok, semboller))


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    main()
