"""capacity_data_sweep.py — P2-11: kapasite VERİYLE mi GÜRÜLTÜYLE mi sınırlı?

Denetim (2026-07-26) önerisi 11: *"Deep model'i 97k/234k satırlık dataset'lerde yeniden
eğit; kapasite merdiveninin gerçekten veriyle mi yoksa gürültüyle mi sınırlı olduğunu
ayır."* Ayrıca "eksikler" §7: 234k ve 97k satırlık setler diskte dururken `deep_mlp`
12k satırlık sette eğitilmişti.

NEDEN TEK BİR BÜYÜK EĞİTİM YETMEZ:
Büyük sette bir kez eğitip "AUC yine 0.50" demek iki hipotezi AYIRMAZ. Ayrım için
**öğrenme eğrisi** gerekir — aynı kapasiteyi ARTAN n'de eğitip OOS-AUC'nin eğimine bak:

  * AUC n ile YÜKSELİYOR            → VERİ sınırlı (daha çok veri işe yarar)
  * AUC n'den bağımsız ~0.50 DÜZ    → GÜRÜLTÜ sınırlı (sinyal yok; veri eklemek çözmez)
  * AUC n ile DÜŞÜYOR               → küçük n'deki yüksek değerler ŞANS/overfit'ti

Bu, "daha çok veri toplayalım mı yoksa sinyal ailesini mi değiştirelim" sorusunun tek
dürüst ayracıdır (ACI DERS #5'in doğrudan sınanması).

YÖNTEM (mevcut makineyi yeniden kullanır — yeni kanıt kapısı icat edilmez):
- `train_neural.load_real_dataset` + `neutralize_degenerate` (σ≈0 bölme mayınları)
- `train_neural.run_cv` = purged+embargolu K-fold (López de Prado Ch.7) + DSR raporu
  (`excess_pseudo_returns` ile — ACI DERS #8 düzeltmesi dahil)
- Alt-örnekleme **KRONOLOJİK OLARAK BAŞTAN** yapılır (`--subsample-mode head`, varsayılan):
  rastgele alt-örnek zaman yapısını bozar ve purge/embargo'yu anlamsızlaştırır. Böylece
  her n noktası "elimde o kadar veri OLSAYDI" sorusunun dürüst karşılığıdır.

HİÇBİR DOSYA YAZILMAZ (--json verilmedikçe) ve HİÇBİR KART KAYDEDİLMEZ: bu bir ÖLÇÜM
script'idir. Kart kaydı `train_neural --save --register`'ın işidir; buradan kaydedilen bir
araştırma kartı canlı köprüyü çalabilirdi (bkz. neural_shadow ARALIK TERCİHİ kilidi).

Kullanım (repo kökünden, -m ile):
    python -m research.capacity_data_sweep --dataset data/meta_dataset_extra_us_...csv
    python -m research.capacity_data_sweep --dataset ... --points 12000,50000,120000,234173
    python -m research.capacity_data_sweep --dataset ... --tier deep_mlp --folds 3 --max-epochs 40
    python -m research.capacity_data_sweep --dataset ... --tier quant_transformer --device auto

`--device` (Tur-86): `fit()`'e reçete olarak geçer; varsayılan `cpu` = eski davranış. Bayrak
yokken transformer basamakları yalnız CPU'da eğitilebiliyordu (5,4M: ~97 örnek/sn → 234k
satırda epoch başına ~40 dk) ve "büyük model veriyle bir şey bulur mu?" sorusu ölçülemiyordu.
İSTENEN aygıt ile GERÇEKLEŞEN aygıt AYRI alanlara yazılır (Tur-42: `auto` CUDA yoksa SESSİZCE
CPU'ya düşer; düşüş, sonucu okuyana görünür olmalı).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

from research.train_neural import load_real_dataset, neutralize_degenerate, run_cv


def sweep(csv_path: str, *, points: list[int], tier: str, folds: int, max_epochs: int,
          purge: int | None = None, device: str = "cpu", optimizer: str = "adamw",
          amp: bool = False, batch_size: int | None = None) -> dict:
    """Her n için purged CV koş, OOS-AUC eğrisini döndür.

    `optimizer`/`amp`/`batch_size` (Tur-86): `fit`'in Tur-42 bellek reçetesi CLI'dan
    erişilebilir olmalı — üst basamaklar (≥282M) 8 GB'lık bir kartta yalnız Adafactor + bf16
    ve küçük toplu-iş ile eğitilebilir. Varsayılanlar `fit` ile AYNI → eski koşular bit-denk."""
    from nihai.ai.neural import _resolve_device

    X, y, feature_names, auto_purge = load_real_dataset(csv_path)
    X = neutralize_degenerate(X, feature_names)
    n_total = len(y)
    purge = purge if purge is not None else auto_purge
    aygit = str(_resolve_device(device))
    recete: dict[str, Any] = {"device": device, "optimizer_name": optimizer, "amp": amp}
    if batch_size:
        recete["batch_size"] = int(batch_size)

    kullanilan = sorted({min(int(p), n_total) for p in points if int(p) >= 500})
    out: dict = {"dataset": csv_path, "n_total": n_total, "tier": tier, "folds": folds,
                 "max_epochs": max_epochs, "purge": purge, "device": device,
                 "device_resolved": aygit, "recete": recete, "noktalar": []}

    print(f"Dataset: {csv_path}")
    print(f"  toplam satır={n_total:,} | özellik={len(feature_names)} | "
          f"taban oranı %{100 * float(y.mean()):.1f} | purge={purge}")
    print(f"  tier={tier} folds={folds} max_epochs={max_epochs} aygıt={device}→{aygit}")
    print(f"  n noktaları: {', '.join(f'{k:,}' for k in kullanilan)}")
    print("=" * 74)

    for n in kullanilan:
        # KRONOLOJİK baştan dilim: veri zaten chrono_sort'lu (load_real_dataset) →
        # rastgele örnekleme purge/embargo'yu anlamsızlaştırırdı.
        Xi, yi = X[:n], y[:n]
        if len(np.unique(yi)) < 2:
            print(f"  n={n:>7,}: ATLANDI (tek sınıf)")
            continue
        _vram_sifirla(aygit)
        t0 = time.perf_counter()
        res = run_cv(Xi, yi, feature_names, tier=tier, folds=folds,
                     max_epochs=max_epochs, purge=purge, recipe=recete)
        sure_s, zirve_mb = round(time.perf_counter() - t0, 1), _vram_zirve(aygit)
        dsr = res.get("dsr_report") or {}
        kayit: dict[str, Any] = {
            "n": n, "taban_orani": round(float(yi.mean()), 4),
            "auc_mean": res.get("auc_mean"), "auc_std": res.get("auc_std"),
            "n_folds_done": res.get("n_folds_done"),
            "dsr": dsr.get("dsr"), "psr": dsr.get("psr"), "sharpe": dsr.get("sharpe"),
            # MALİYET de ölçümün parçası: bir basamağın "daha iyi" olup olmadığı, neye mal
            # olduğu bilinmeden okunamaz (Tur-67: amortisman kararı verir).
            "param_count": res.get("param_count"), "sure_s": sure_s,
            "brier_mean": res.get("brier_mean"), "ece_mean": res.get("ece_mean"),
            "zirve_vram_mb": zirve_mb}
        out["noktalar"].append(kayit)
        auc = kayit["auc_mean"]
        print(f"  n={n:>7,}  taban=%{100 * kayit['taban_orani']:.1f}  "
              f"OOS-AUC={auc:.4f} ± {kayit['auc_std']:.4f}  DSR={kayit['dsr']}")

    out["verdikt"] = _verdikt(out["noktalar"])
    return out


def _vram_sifirla(aygit: str) -> None:
    """CUDA zirve sayacını sıfırla (CPU'da NO-OP)."""
    if not str(aygit).startswith("cuda"):
        return
    import torch
    torch.cuda.reset_peak_memory_stats()


def _vram_zirve(aygit: str) -> float | None:
    """Bu nokta boyunca ayrılan zirve VRAM (MB); CPU'da None — 0.0 YAZILMAZ (Tur-31:
    ölçüm yokluğu ölçüm gibi raporlanamaz)."""
    if not str(aygit).startswith("cuda"):
        return None
    import torch
    return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)


def _egim(noktalar: list[dict]) -> float:
    """log10(n) ↔ OOS-AUC en küçük kareler eğimi (AUC puanı / veri dekadı)."""
    xs = [float(np.log10(p["n"])) for p in noktalar]
    ys = [float(p["auc_mean"]) for p in noktalar]
    mx, my = float(np.mean(xs)), float(np.mean(ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    return sum((x - mx) * (yv - my) for x, yv in zip(xs, ys, strict=True)) / sxx


def _verdikt(noktalar: list[dict], *, dsr_esik: float = 0.05) -> dict:
    """Öğrenme eğrisinden hüküm — VERİ mi GÜRÜLTÜ mü sınırlıyor?

    ⚠️ TASARIM DERSİ (2026-07-31, bu fonksiyonun İLK sürümü bozuktu):
    İlk sürüm hükmü YALNIZ log-doğrusal eğime dayandırıyordu. Gerçek koşularda bu
    yanıltıcı çıktı — çünkü 5 noktalı bir eğride EN KÜÇÜK n'nin gürültüsü (hata payı
    orada en büyük) doğruyu tek başına döndürüyor:
      * 97k PIT setinde ilk nokta YÜKSEKTİ  → "küçük n şanstı" etiketi
      * 234k setinde ilk nokta DÜŞÜKTÜ      → "veri sınırlı" etiketi
    …oysa İKİSİNDE DE 25k sonrası eğri DÜZ ve **DSR her n'de ≈ 0** idi. Yani etiket
    raporun geri kalanıyla ÇELİŞİYORDU = ACI DERS #8'in aynısı (kapı bozuksa sonuç değil
    kapı düzeltilir).

    Düzeltilmiş sıra — ÖNCE kanıt kapısı, SONRA eğim:
      1. Hiçbir n'de DSR > `dsr_esik` yoksa → `KENAR_YOK_HER_N`. Bu BASKIN gerçektir;
         AUC eğimi tartışmak anlamsızdır (deflasyondan sonra beceri kanıtı yok).
      2. DSR bir yerde kapıyı geçiyorsa eğime bakılır — ve eğim İKİ KEZ hesaplanır:
         tüm noktalarla VE en küçük n atılarak. İkisi AYNI hükmü vermiyorsa eğri tek bir
         uç noktaya bağımlıdır → `EGRI_KARARSIZ` (uydurma yapma, nokta ekle).
    """
    kullanilir = [p for p in noktalar if isinstance(p.get("auc_mean"), (int, float))]
    if len(kullanilir) < 3:
        return {"label": "YETERSIZ_NOKTA",
                "text": f"{len(kullanilir)} n noktası — eğim için en az 3 gerekir."}

    ys = [float(p["auc_mean"]) for p in kullanilir]
    egim_tum = _egim(kullanilir)
    egim_kirpik = _egim(kullanilir[1:]) if len(kullanilir) >= 4 else None
    dsrler = [d for p in kullanilir if isinstance(d := p.get("dsr"), (int, float))]
    dsr_max = max(dsrler) if dsrler else None
    en_buyuk = kullanilir[-1]

    out = {"egim_per_dekad": round(egim_tum, 4),
           "egim_ilk_nokta_haric": (round(egim_kirpik, 4) if egim_kirpik is not None else None),
           "auc_ilk": round(ys[0], 4), "auc_son": round(ys[-1], 4),
           "auc_en_iyi": round(max(ys), 4), "dsr_max": dsr_max, "dsr_esik": dsr_esik,
           "dsr_olculen_nokta": len(dsrler),
           "en_buyuk_n": en_buyuk["n"], "en_buyuk_dsr": en_buyuk.get("dsr")}

    # 1) KANIT KAPISI ÖNCE — deflasyondan sonra hiçbir n'de beceri yoksa eğim tartışması yok.
    if dsr_max is None:
        # Tur-86: "max DSR=None ≤ 0,05" ölçülemeyeni eşikle kıyaslanmış gibi basıyordu.
        # `dsr_report` None = fazlalık serisi sıfır varyanslı: model HER test örneğinde
        # çoğunluk kuralıyla aynı kararı verdi → beceri tanım gereği YOK (hüküm aynı, gerekçe
        # farklı; BULGU-86-2'nin kardeşi).
        out["label"] = "KENAR_YOK_HER_N"
        out["text"] = (f"DSR hiçbir n'de HESAPLANAMADI: fazlalık serisi sıfır varyanslı — model "
                       f"hiçbir test örneğinde çoğunluk kuralından AYRILMADI → tanım gereği beceri "
                       f"YOK. AUC {min(ys):.3f}–{max(ys):.3f} bandında; sınırlayan veri MİKTARI da "
                       f"model KAPASİTESİ de değil, sinyalin kendisi (ACI DERS #5).")
        return out
    if dsr_max <= dsr_esik:
        out["label"] = "KENAR_YOK_HER_N"
        out["text"] = (f"Hiçbir n'de DSR kapıyı geçmiyor (max DSR={dsr_max} ≤ {dsr_esik}) → "
                       f"deflasyondan sonra HİÇBİR veri boyutunda beceri kanıtı yok. "
                       f"AUC {min(ys):.3f}–{max(ys):.3f} bandında salınıyor; sınırlayan veri "
                       f"MİKTARI da model KAPASİTESİ de değil, sinyalin kendisi (ACI DERS #5). "
                       f"Bu sinyal ailesinde veri toplamaya devam etmenin ölçülmüş gerekçesi YOK.")
        return out

    # 2) DSR bir yerde geçiyorsa eğim — ama tek uç noktaya bağımlıysa hüküm VERME.
    def _etiket(e: float) -> str:
        return "VERI_SINIRLI" if e >= 0.010 else ("KUCUK_N_SANSIYDI" if e <= -0.010
                                                  else "GURULTU_SINIRLI")

    et_tum = _etiket(egim_tum)
    if egim_kirpik is not None and _etiket(egim_kirpik) != et_tum:
        out["label"] = "EGRI_KARARSIZ"
        out["text"] = (f"Eğim tüm noktalarla {egim_tum:+.4f}, en küçük n atılınca "
                       f"{egim_kirpik:+.4f} → hüküm TEK bir uç noktaya bağımlı. Eğri kararsız; "
                       f"ara n noktaları ekleyip tekrar ölç (uydurma hüküm verme).")
        return out

    out["label"] = et_tum
    if et_tum == "VERI_SINIRLI":
        out["text"] = (f"OOS-AUC her 10× veride {egim_tum:+.4f} artıyor (ilk nokta hariç de "
                       f"aynı yön) ve DSR bir n'de kapıyı geçiyor (max={dsr_max}) → daha çok "
                       f"veri İŞE YARIYOR.")
    elif et_tum == "KUCUK_N_SANSIYDI":
        out["text"] = (f"OOS-AUC veri arttıkça DÜŞÜYOR ({egim_tum:+.4f}/dekad, ilk nokta hariç "
                       f"de aynı yön) → küçük n'deki yüksek değerler şans/overfit idi.")
    else:
        out["text"] = (f"OOS-AUC n'den BAĞIMSIZ ({egim_tum:+.4f}/dekad) → sınırlayan veri "
                       f"MİKTARI değil, sinyalin kendisi (ACI DERS #5).")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="P2-11: öğrenme eğrisi — kapasite veriyle mi gürültüyle mi sınırlı?")
    ap.add_argument("--dataset", required=True, help="gerçek CSV (FEATURE_NAMES+label şeması)")
    ap.add_argument("--points", default=None,
                    help="virgüllü n listesi (vars: ~geometrik 5 nokta, n_total'e kadar)")
    ap.add_argument("--tier", default="deep_mlp")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--purge", type=int, default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                    help="eğitim aygıtı (train_neural ile aynı seçenekler; vars. cpu = eski davranış)")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"],
                    help="Tur-42 bellek reçetesi; üst basamaklar 8 GB'ta adafactor ister")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (yalnız CUDA'da etkili)")
    ap.add_argument("--batch-size", type=int, default=None, help="fit toplu-iş (vars. 256)")
    ap.add_argument("--json", default=None, help="sonucu bu yola yaz (verilmezse dosya YAZILMAZ)")
    args = ap.parse_args(argv)

    if args.points:
        points = [int(p) for p in args.points.split(",") if p.strip()]
    else:
        # geometrik 5 nokta: 12k'dan toplam boyuta kadar (12k = mevcut kartın taban n'i)
        _X, y, _fn, _p = load_real_dataset(args.dataset)
        n_total = len(y)
        points = [int(round(12000 * (n_total / 12000) ** (i / 4))) for i in range(5)]

    res = sweep(args.dataset, points=points, tier=args.tier, folds=args.folds,
                max_epochs=args.max_epochs, purge=args.purge, device=args.device,
                optimizer=args.optimizer, amp=args.amp, batch_size=args.batch_size)

    print("=" * 74)
    v = res["verdikt"]
    print(f"HÜKÜM [{v.get('label')}]: {v.get('text')}")
    print("Not: bu bir KAPASİTE/veri ölçümüdür — kenar iddiası İÇERMEZ (ACI DERS #3/#4).")

    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
