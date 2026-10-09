"""
iv_rank_threshold_test.py — `iv_rank_low/high_threshold` (30/50) eşiklerini ÖLÇER.

ÖLÇÜLEN İDDİA (kodun gerçekte yaptığı, varsayılan değil):
`strategy.signal_quality_score` satır 910-915 IV Rank'i "altın bant" olarak kullanır —
`30 <= iv_rank <= 50` ise skora **+0.5** ekler, yorumu: "ne çok düşük ne çok yüksek =
uzun opsiyon için ideal". Yani sınanacak önerme "yüksek IV → prim sat" DEĞİL:

    H: IV Rank [30,50] bandındaki kurulumlar UZUN OPSİYON için bandın dışındakilerden
       daha iyidir.

VEKİL ÖLÇÜT: uzun opsiyon pozitif gamma taşır; ödeme yapması için gerçekleşen
oynaklığın örtük oynaklığı AŞMASI gerekir. Dolayısıyla
    uzun_opsiyon_kenari(t) = hv_ileri(t) − iv_current(t)
Pozitifse uzun opsiyon lehine. `hv_ileri`, W gün ileri kaydırılmış `hv_current`'tır;
W sağlayıcı tarafından belgelenmediği için TEK BİR DEĞER SEÇİLMEZ, `--hv-window` ile
taranır ve sonuç W'ye duyarlılık olarak raporlanır (aşağıdaki HV PENCERESİ notuna bak).

KANIT KAPILARI (proje disiplini):
- ACI DERS #9: bant, KOŞULSUZ TABANLA kıyaslanır. "Bant içi ortalama pozitif" tek
  başına hiçbir şey demez; taban zaten pozitifse bant bilgi eklemiyordur.
- ACI DERS #7: `--exec-lag` — rank t-1'den okunur, kenar t'de ölçülür.
- Tur-23 kuralı: kapı PASS dese bile SEÇİCİLİK (n_kept oranı) okunur.
- ACI DERS #8 (t şişmesi) İKİ kanaldan kapatılır: `--non-overlap` (ardışık günlerin
  ileri pencereleri örtüşür) ve gün-bazlı eşleştirilmiş t (aynı gün tüm semboller
  birlikte oynar). VERDİKT gün-bazlı t'ye bakar, ham Welch'e değil.
- Welch/t eşiği ±1,645 (Tur-24 ile aynı).

⚠️ TUR-51 UYARISI — BU SCRIPT'İN VERDİKTİ ARTIK KARAR MERCİİ DEĞİLDİR.
Vekil etiket (`hv_current.shift(-W)`) iki kez ölçülerek çürütüldü (BULGU-51-1/51-3;
ayrıntı aşağıdaki "HV PENCERESİ" bloğunda). Bu script'in 2026-08-13'te ürettiği
"bant TERS + ilişki MONOTON (t=−7,35)" sonucu VEKİLİN ARTEFAKTIDIR: fiyattan
hesaplanan temiz ileri RV ile AYNI satırlarda bant t=+0,61, düzenli ızgarada t=−0,17
= AYIRT EDİLEMEZ. Eşik/bonus kararı için `research/edge_lab/build_vrp_dataset.py`
+ `docs/EDGE_FINDINGS.md` §28 okunmalıdır.

Kullanım (ÖNERİLEN tam biçim):
    python -m research.iv_rank_threshold_test --non-overlap --exec-lag 1 --sweep
    python -m research.iv_rank_threshold_test --non-overlap --hv-window 42 --sweep
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nihai.data import dolt_options

ROLLING = 252          # iv_history.ROLLING_WINDOW_DAYS ile aynı
T_ESIK = 1.645         # Tur-24 Welch eşiği


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """a'nın b'den farkı için Welch t. Örneklem yetersizse nan."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


# ⚠️ TUR-51: BU SCRIPT'İN VEKİL ETİKETİ ÖLÇÜLDÜ ve VERDİKTİ TAŞIMIYOR.
# Aşağıdaki "W taranmalı" gerekçesi ARTIK GEÇERSİZDİR — pencere ÖLÇÜLDÜ (BULGU-51-1):
# `hv_current` geriye-bakan **30 işlem günlük** RV'dir (568.195 satır / 603 sembol,
# kor. 0,99504; sembol ortancası 0,9998; likiditeden bağımsız). İki sonucu var:
#   (1) `hv_current(t+W)` (t+W−30, t+W] ölçer → W<30 iken etiketin (30−W)/30'u KARAR
#       BARINDAN ÖNCEDİR. Eski varsayılan W=21'de bu oran **%30**'du.
#   (2) Daha ağırı (BULGU-51-3): dolt serisi DÜZENSİZDİR (fiyat barlarının ~%93,4'ü,
#       boşluklar ÖBEKLİ) → `shift(-30)` hedeflenen 42-44 takvim günü yerine GERÇEKTE
#       ortanca **70**, ortalama 65,2, std **34,0** gün ileri gider. Ufuk sembole ve
#       döneme göre DEĞİŞİR; bu bir sabit-ufuk ölçümü DEĞİLDİR.
# AYNI SATIRLARDA İZOLE EDİLDİ (aynı evren/dönem/W=30/rank/exec-lag/örtüşmesiz; tek
# değişken ETİKET): vekil bant t=**−2,29** + monoton dilim deseni ↔ fiyattan hesaplanan
# temiz etiket t=**+0,61**, monotonluk YOK; iki etiketin korelasyonu yalnız **0,7177**.
# Yani "bant TERS, ilişki MONOTON" bulgusu VEKİLİN ARTEFAKTIDIR, oynaklığın özelliği değil.
# DOĞRU ÖLÇÜM: `research/edge_lab/build_vrp_dataset.py` (ileri RV fiyattan, düzenli ızgara).
# Bu script TEŞHİS amacıyla duruyor; verdikt satırını KENAR/EŞİK kararı için OKUMA.
#
# HV PENCERESİ — neden tahmin edilmiyor, TARANIYOR:
# İlk sürüm `hv_month_ago(t) ≈ hv_current(t−L)` uydurmasıyla L'yi "ampirik" buluyordu
# ve L=15 döndürüyordu. İki yönden yanlıştı: (1) o hesap HV penceresini değil "bir ay
# öncesi" OFSETİNİ arar — bana gereken pencere W'dir, çünkü hv_current(t+W) = (t, t+W]
# aralığının gerçekleşen oynaklığıdır; (2) bulunan L arama aralığının SINIRINDAYDI
# (15..35 arasında 15), yani gerçek asgari aralığın dışındaydı = uydurma sonuç.
# Sağlayıcı W'yi belgelemiyor, o yüzden dürüst yol tek bir W seçmek değil, sonucu
# W'ye karşı DUYARLILIK olarak raporlamaktır (--hv-window ile taranır).
VARSAYILAN_HV = 30          # TUR-51: ÖLÇÜLDÜ — kirlenmenin sıfır olduğu TEK W


def veri_hazirla(symbols: list[str] | None, min_days: int, exec_lag: int,
                 hv_window: int | None = None, non_overlap: bool = False) -> pd.DataFrame:
    """Yerel toplu CSV → sembol başına üretim-eşdeğeri IV Rank + ileri HV kenarı."""
    yol = dolt_options.bulk_csv("volatility_history")
    if yol is None:
        raise SystemExit("toplu CSV yok — önce `python -m research.backfill_iv_history` koşun")
    kolonlar = ["date", "act_symbol", "iv_current", "hv_current", "hv_month_ago"]
    df = dolt_options.read_bulk(yol, cols=kolonlar, symbols=symbols or None)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "iv_current", "hv_current"])
    df = df.sort_values(["act_symbol", "date"], ignore_index=True)
    print(f"ham: {len(df):,} satır / {df['act_symbol'].nunique():,} sembol "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    L = hv_window or VARSAYILAN_HV
    print(f"HV/ileri pencere W={L} işlem günü (sonuç buna karşı taranmalı)")

    g = df.groupby("act_symbol", sort=False)["iv_current"]
    lo = g.transform(lambda s: s.rolling(ROLLING, min_periods=min_days).min())
    hi = g.transform(lambda s: s.rolling(ROLLING, min_periods=min_days).max())
    span = hi - lo
    # get_iv_rank ile AYNI matematik: (iv − min) / (max − min) × 100, [0,100] kırpma
    df["iv_rank"] = np.where(span > 1e-9, (df["iv_current"] - lo) / span * 100.0, np.nan)
    df["iv_rank"] = df["iv_rank"].clip(0, 100)

    # ACI DERS #7: karar rank'i exec_lag bar ÖNCEDEN okunur
    if exec_lag:
        df["iv_rank"] = df.groupby("act_symbol", sort=False)["iv_rank"].shift(exec_lag)

    # hv_ileri: L gün SONRAKİ hv_current = (t, t+L] aralığının gerçekleşen oynaklığı
    df["hv_ileri"] = df.groupby("act_symbol", sort=False)["hv_current"].shift(-L)
    df["kenar"] = df["hv_ileri"] - df["iv_current"]
    df = df.dropna(subset=["iv_rank", "kenar"])

    # ÖRTÜŞME: ardışık günlerin ileri pencereleri neredeyse aynı aralığı ölçer →
    # gözlemler bağımsız DEĞİL, t istatistiği şişer (ACI DERS #8'in kalıbı: kapı
    # raporun geri kalanıyla çelişecek kadar iyi çıkar). Sembol başına her L. satırı
    # alarak örtüşmeyi kaldırırız; n düşer ama t DÜRÜST olur.
    if non_overlap:
        df = df[df.groupby("act_symbol", sort=False).cumcount() % L == 0]
        print(f"örtüşme kaldırıldı (sembol başına her {L}. gözlem)")
    return df


def _gun_bazli_t(df: pd.DataFrame, maske: pd.Series) -> tuple[float, int]:
    """Gün-bazlı EŞLEŞTİRİLMİŞ t: (kova ortalaması − o günün taban ortalaması).

    Örtüşme kaldırılsa bile aynı gün TÜM sembollerin oynaklığı birlikte oynar
    (kesitsel korelasyon) → gözlemler hâlâ bağımsız değil ve t şişer. Her günü tek
    gözleme indirip tabanla eşleştirmek bu kanalı tamamen kapatır; n gün sayısına
    düşer ama kalan anlamlılık piyasa-geneli hareketten DEĞİL, kovadan gelir.
    """
    gun_kova = df[maske].groupby("date")["kenar"].mean()
    gun_taban = df.groupby("date")["kenar"].mean()
    fark = (gun_kova - gun_taban).dropna()
    if len(fark) < 3:
        return float("nan"), len(fark)
    sd = fark.std(ddof=1)
    if sd <= 0:
        return float("nan"), len(fark)
    return float(fark.mean() / (sd / np.sqrt(len(fark)))), len(fark)


def rapor(df: pd.DataFrame, lo_esik: float, hi_esik: float) -> dict:
    kenar = df["kenar"].to_numpy()
    ic_maske = (df["iv_rank"] >= lo_esik) & (df["iv_rank"] <= hi_esik)
    ic, dis = kenar[ic_maske.to_numpy()], kenar[~ic_maske.to_numpy()]
    n = len(kenar)
    secicilik = len(ic) / n * 100 if n else 0.0

    print(f"\n{'='*66}")
    print(f"BANT [{lo_esik:g}, {hi_esik:g}] — uzun opsiyon kenarı = hv_ileri − iv")
    print(f"{'='*66}")
    print(f"  KOŞULSUZ TABAN   n={n:>8,}  ort={kenar.mean():+.5f}")
    print(f"  bant İÇİ         n={len(ic):>8,}  ort={ic.mean():+.5f}   (seçicilik %{secicilik:.1f})")
    print(f"  bant DIŞI        n={len(dis):>8,}  ort={dis.mean():+.5f}")
    t_dis = _welch_t(ic, dis)
    t_taban = _welch_t(ic, kenar)
    t_gun, n_gun = _gun_bazli_t(df, ic_maske)
    print(f"\n  Welch t (içeri vs dışarı) = {t_dis:+.2f}")
    print(f"  Welch t (içeri vs TABAN)  = {t_taban:+.2f}   <- ACI DERS #9 kapısı")
    print(f"  GÜN-BAZLI eşleştirilmiş t = {t_gun:+.2f}  (n={n_gun} gün)  <- kesitsel korelasyon giderilmiş")
    gecti = bool(np.isfinite(t_gun) and t_gun > T_ESIK)
    print(f"\n  VERDİKT: {'BANT BİLGİ TAŞIYOR' if gecti else 'KAPI GEÇİLEMEDİ — bant bilgi taşımıyor'}")
    return {"t_dis": t_dis, "t_taban": t_taban, "t_gun": t_gun,
            "gecti": gecti, "secicilik": secicilik}


def sweep(df: pd.DataFrame) -> None:
    """Herhangi BİR bant ayırıyor mu? Onluk dilim taraması."""
    kenar = df["kenar"].to_numpy()
    rank = df["iv_rank"].to_numpy()
    print(f"\n{'='*66}")
    print(f"ONLUK DİLİM TARAMASI (taban ort = {kenar.mean():+.5f})")
    print(f"{'='*66}")
    print(f"  {'dilim':<12} {'n':>9} {'ort kenar':>12} {'t vs taban':>12} {'GÜN-BAZLI t':>13}")
    for a in range(0, 100, 10):
        m = (rank >= a) & (rank < a + 10)
        alt = kenar[m]
        if len(alt) < 30:
            continue
        t_gun, _ = _gun_bazli_t(df, pd.Series(m, index=df.index))
        print(f"  [{a:>3},{a+10:>3})   {len(alt):>9,} {alt.mean():>+12.5f} "
              f"{_welch_t(alt, kenar):>+12.2f} {t_gun:>+13.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="IV Rank eşiklerini ölç")
    ap.add_argument("--symbols", default=None, help="virgülle ayrık; boş = TÜM evren")
    ap.add_argument("--min-days", type=int, default=252,
                    help="rank için gereken asgari gözlem (üretim 30'da PARTIAL verir)")
    ap.add_argument("--exec-lag", type=int, default=0, help="ACI DERS #7 duyarlılığı")
    ap.add_argument("--low", type=float, default=None)
    ap.add_argument("--high", type=float, default=None)
    ap.add_argument("--sweep", action="store_true", help="onluk dilim taraması")
    ap.add_argument("--hv-window", type=int, default=None,
                    help="HV penceresini elle ver (duyarlılık kontrolü)")
    ap.add_argument("--non-overlap", action="store_true",
                    help="örtüşmeyen örneklem — t şişmesini kaldırır (ÖNERİLEN)")
    args = ap.parse_args()

    from nihai.config import CONFIG
    lo = args.low if args.low is not None else CONFIG.strategy.iv_rank_low_threshold
    hi = args.high if args.high is not None else CONFIG.strategy.iv_rank_high_threshold

    syms = ([s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None)
    df = veri_hazirla(syms, args.min_days, args.exec_lag,
                      hv_window=args.hv_window, non_overlap=args.non_overlap)
    print(f"ölçülebilir: {len(df):,} satır (exec_lag={args.exec_lag}, "
          f"min_days={args.min_days}, ortusmeyen={args.non_overlap})")

    rapor(df, lo, hi)
    if args.sweep:
        sweep(df)
    return 0


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    raise SystemExit(main())
