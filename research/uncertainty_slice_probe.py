"""Tur-77 · Madde 5.1/5.2 ölçümü — belirsizlik ölçüleri isabeti DOĞRU YÖNDE sıralıyor mu?

Listenin karar kuralı *"epistemik risk yüksekse ABSTAIN"* bir YÖN iddiasıdır: yüksek
belirsizlikte model daha çok yanılmalıdır. Tur-36 aynı soruyu MC-dropout std'si için sordu ve
canlıda TERS çıktı (0/5 tohum). Burada ayrışmanın her bileşeni aynı protokolle sınanır:

  canlı 1m seti · KRONOLOJİK %70/%30 bölme (arada purge boşluğu) · N tohum ·
  ölçü başına: EN DÜŞÜK belirsizlik çeyreğinin isabeti − EN YÜKSEK çeyreğin isabeti
  (pozitif = doğru yön; "yüksekte daha çok yanılıyor").

KARAR-DIŞI araştırma aracıdır; hiçbir artefakt/model kaydetmez, yalnız basar.

    python -m research.uncertainty_slice_probe [--dataset data/meta_dataset_live_1m_2026-07-30.csv]
                                              [--seeds 5] [--members 5] [--T 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nihai.console import ensure_utf8_stdout

VARSAYILAN_SET = Path("data") / "meta_dataset_live_1m_2026-07-30.csv"


def dilim_farki(belirsizlik: np.ndarray, isabet: np.ndarray) -> float | None:
    """Düşük-belirsizlik çeyreği isabeti − yüksek-belirsizlik çeyreği isabeti.
    Bağlar kararlı sıralamayla (mergesort) çözülür; <8 örnek → None."""
    n = belirsizlik.size
    if n < 8 or isabet.size != n:
        return None
    sira = np.argsort(belirsizlik, kind="mergesort")
    k = n // 4
    return float(np.mean(isabet[sira[:k]]) - np.mean(isabet[sira[-k:]]))


def olc(dataset: Path, seeds: int, members: int, T: int, max_epochs: int) -> dict:
    from nihai.ai import neural
    from research.train_neural import load_real_dataset

    X, y, names, purge = load_real_dataset(str(dataset))
    n = len(y)
    kesim = int(n * 0.7)
    bosluk = int(purge or 0)
    Xtr, ytr = X[:kesim], y[:kesim]
    Xte, yte = X[kesim + bosluk:], y[kesim + bosluk:]
    sonuc: dict[str, list[float]] = {}
    auc: list[float] = []

    def ekle(ad: str, deger: np.ndarray, isabet: np.ndarray) -> None:
        d = dilim_farki(np.asarray(deger, dtype=float), isabet)
        if d is not None:
            sonuc.setdefault(ad, []).append(d)

    for s in range(seeds):
        tek = neural.NeuralCapacity("deep_mlp", feature_names=names, seed=s,
                                    extra_heads=("dir_evid",))
        tek.fit(Xtr, ytr, max_epochs=max_epochs, patience=5)
        d = tek.predict_uncertainty_decomposition(Xte, T=T)
        b = tek.predict_belief(Xte)
        if d is None:
            continue
        isabet = ((d["p"] >= 0.5).astype(float) == (yte >= 0.5).astype(float)).astype(float)
        auc_s = neural.evaluation_metrics(yte, d["p"])["auc"]
        if auc_s is not None:
            auc.append(auc_s)
        ekle("tek_mc_karsilikli_bilgi", d["karsilikli_bilgi"], isabet)
        ekle("tek_mc_epistemik_var", d["epistemik_var"], isabet)
        ekle("tek_aleatorik_entropi", d["aleatorik_entropi"], isabet)
        ekle("guvensizlik_|p-0.5|_ters", -np.abs(d["p"] - 0.5), isabet)
        if b is not None:
            ekle("edl_bosluk_u", b["bosluk"], isabet)

        ens = neural.NeuralEnsemble("deep_mlp", feature_names=names, seed=1000 + s,
                                    n_members=members)
        ens.fit(Xtr, ytr, max_epochs=max_epochs, patience=5)
        de = ens.predict_uncertainty_decomposition(Xte, T=T)
        if de is None:
            continue
        isabet_e = ((de["p"] >= 0.5).astype(float) == (yte >= 0.5).astype(float)).astype(float)
        ekle("topluluk_uye_arasi_var", de["uye_arasi_var"], isabet_e)
        ekle("topluluk_karsilikli_bilgi", de["karsilikli_bilgi"], isabet_e)
        ekle("topluluk_aleatorik_entropi", de["aleatorik_entropi"], isabet_e)

    return {
        "dataset": str(dataset), "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "purge": bosluk, "seeds": seeds, "members": members, "T": T,
        "test_auc_ort": (round(float(np.mean(auc)), 4) if auc else None),
        "olculer": {ad: {"fark_ort": round(float(np.mean(v)), 4),
                         "dogru_yon": f"{int(np.sum(np.asarray(v) > 0))}/{len(v)}",
                         "tohumlar": [round(x, 4) for x in v]}
                    for ad, v in sonuc.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Belirsizlik ölçüsü dilim-isabeti sondası (KARAR-DIŞI)")
    ap.add_argument("--dataset", default=str(VARSAYILAN_SET))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--max-epochs", type=int, default=30, dest="max_epochs")
    a = ap.parse_args(argv)
    print(json.dumps(olc(Path(a.dataset), a.seeds, a.members, a.T, a.max_epochs),
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
