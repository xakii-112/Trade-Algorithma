"""Tur-67 — Monte Carlo yollarının vektörleştirilmesi (katalog #21) ve BİT-DENKLİK kilidi.

Tur-66 kataloğun asenkron kümesini kapattı; bu tur **ağır matematik** kümesini profille
açtı. Profil iki saf-Python döngüsü gösterdi ve ikisi de vektörleştirildi:

| fonksiyon | ÖNCE | SONRA | oran |
|---|---|---|---|
| `backtest.monte_carlo_paths` (200 işlem × 1000 yin.) | 180,05 ms | 70,48 ms | **2,55×** |
| `backtest.monte_carlo_paths` (200 × 5000) | 891,44 ms | 356,26 ms | 2,50× |
| `quant_math.mc_bootstrap_terminal` (n=1500·h=21·4000 yol) | 146,22 ms | **2,11 ms** | **69,2×** |
| `mc_bootstrap_terminal` (n=250·h=63·4000 yol) | 314,42 ms | 5,91 ms | 53,2× |

**Bu turun asıl kuralı BİT-DENKLİKTİR.** İkisi de `seed` alan ve docstring'inde *"aynı
seed → bit-bit aynı çıktı"* sözü veren fonksiyonlardır; bir hızlanma o sözü bozuyorsa
hızlanma değil SESSİZ BİR DAVRANIŞ DEĞİŞİKLİĞİDİR. Tur-65 CUDA'yı (74,7×) tam bu gerekçeyle
opt-in bırakmıştı. Burada denklik ölçülerek sağlandı, varsayılarak değil — ve iki ayrı
tuzak çıktı:

1. **Toplama SIRASI.** İlk aday `cumsum(pnl)` yapıp sonra sermayeyi ekliyordu; eski gövde
   ise sermayeden başlayıp tek tek ekliyor. Ölçüldü: `final` eşit DEĞİL, maks |ΔMDD|
   **8,04e-14**. Sermaye diziye İLK ELEMAN yapılınca sıra birebir oturdu.
2. **Çarpım İNDİRGEMESİ.** `np.prod(..., axis=1)` eksende pairwise indirger; eski gövde
   her yolu ayrı 1-B `np.prod` ile çarpıyordu. `np.multiply.accumulate(...)[:, -1]`
   ardışık çarpar ve sırayı korur.

**RNG akışına DOKUNULMADI.** `monte_carlo_paths` hâlâ `random.Random(seed).shuffle` ile yol
yol karıştırır (ölçüldü: karıştırma toplamın yalnız **%33'ü**; kalan %67 zaten vektörleşiyor,
akışı bozmaya değmez). `mc_bootstrap_terminal`de toplu çekim kullanıldı ÇÜNKÜ ölçüldü:
`rng.integers(0, m, size=N)` ile N skaler çağrı **aynı akışı** verir (5 ayrı `m` için
sınandı) ve satır-öncelikli sıra eski yol-yol çekişle örtüşür.

**Ölçülüp REDDEDİLENLER** (bu dosya onları da kayda geçirir, çünkü bir sonraki tur aynı
maddeleri yeniden açmasın):

* **#15/#81 Numba** — aynı kernelde NumPy'a karşı gerçekten hızlı (`njit` **16,1–21,9×**,
  `prange` **62,4–167,3×**, üstelik bit-denk). Ama görünmeyen maliyeti ölçüldü:
  `import numba` **969,9 ms**, ilk çağrı (JIT) **1211,5 ms**, derlenmiş çağrı 0,133 ms →
  NumPy'ın 2,11 ms'ine karşı **613 çağrıdan sonra** kara geçiyor. `/api/quant/mc` bir
  panoda elle açılan uçtur; süreç yeniden başlamadan 613 çağrıyı görmez. Ayrıca 969,9 ms'lik
  import, Tur-26'nın `import nihai.ai` 3854 → 11 ms dersinin doğrudan geri alınması olurdu.
* **#16 TA-Lib / ham bottleneck** — `pandas rolling(14).mean()` 237,2 µs ↔ `bn.move_mean`
  12,2 µs = **19,45×** fark GERÇEK (pandas 3.0 rolling'i bottleneck'e göndermiyor). Ama
  payı ölçüldü: 2000 barlık backtestte rolling **14 çağrı / 0,003 sn = %0,5**.
  `_SharedIndicators` (Tur-7/43) göstergeleri zaten bir kez hesaplıyor. %0,5 için her
  göstergeyi yeni bir C bağımlılığına taşımak yapılmaz.
* **#14 ciso8601 (*"50 kat"*)** — depo ZATEN hızlı yolda: `datetime.fromisoformat`
  **0,271 µs**. Kataloğun eleştirdiği yollar `pd.to_datetime` **578,2 µs** (2133×) ve
  `strptime` **12,7 µs**. Öncül bu depoda yok.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from nihai.ai import quant_math as qm
from nihai.execution.backtest import max_drawdown, monte_carlo_paths


# --- Referans gövdeler: Tur-67 ÖNCESİ mantığın birebir kopyası -------------------
# Bunlar "eski kod" değil, DENKLİK TANIĞIDIR: vektörel gövde bir gün yeniden yazılırsa
# karşılaştırılacak ölçüt burada durur (yeniden yazan kişi kendi çıktısını ölçüt yapamaz).
def _mc_referans(trade_pnls, initial_equity, n_iters=1000, seed=42):
    if not trade_pnls:
        return {}
    rng = random.Random(seed)
    fe: list[float] = []
    dd: list[float] = []
    iflas = 0
    for _ in range(n_iters):
        order = trade_pnls.copy()
        rng.shuffle(order)
        curve = [initial_equity]
        for pnl in order:
            curve.append(curve[-1] + pnl)
        fe.append(curve[-1])
        dd.append(max_drawdown(curve))
        iflas += min(curve) <= 0    # Tur-95 (BULGU-95-2): iflas YOL İÇİNDE sayılır
    fe.sort()
    dd.sort()

    def pct(arr, p):
        return arr[int(len(arr) * p)]

    return {
        "iterations": n_iters,
        "final_equity_p05": pct(fe, 0.05), "final_equity_p50": pct(fe, 0.50),
        "final_equity_p95": pct(fe, 0.95), "max_drawdown_p05": pct(dd, 0.05),
        "max_drawdown_p50": pct(dd, 0.50), "max_drawdown_p95": pct(dd, 0.95),
        "ruin_probability": iflas / n_iters,
    }


def _boot_referans(returns, horizon, n_paths, seed, block=5):
    empty = {"p5": None, "p50": None, "p95": None, "prob_loss": None}
    arr = qm._clean_1d(returns, min_len=2)
    if arr is None:
        return empty
    h, npaths, blk = int(horizon), int(n_paths), max(1, int(block))
    if h <= 0 or npaths <= 0:
        return empty
    n = arr.size
    rng = np.random.default_rng(seed)
    terminal = np.empty(npaths, dtype=float)
    max_start = max(1, n - blk + 1)
    for p in range(npaths):
        picked: list[float] = []
        while len(picked) < h:
            start = int(rng.integers(0, max_start))
            picked.extend(float(v) for v in arr[start:start + blk])
        terminal[p] = float(np.prod(1.0 + np.asarray(picked[:h], dtype=float))) - 1.0
    return {"p5": float(np.percentile(terminal, 5)), "p50": float(np.percentile(terminal, 50)),
            "p95": float(np.percentile(terminal, 95)),
            "prob_loss": float(np.mean(terminal < 0.0))}


# --- 1. monte_carlo_paths BİT-DENK -------------------------------------------------
@pytest.mark.parametrize(("n_tr", "iters", "eq", "seed"), [
    (60, 300, 100_000.0, 42),
    (200, 200, 100_000.0, 42),
    (1, 200, 50_000.0, 7),        # tek işlem: MDD ya 0 ya tek düşüş
    (2, 150, 10_000.0, 1),
    (30, 200, 1_000.0, 5),        # küçük sermaye → MDD yüzdesi büyük
])
def test_mc_paths_bit_denk(n_tr, iters, eq, seed):
    pnls = list(np.random.default_rng(11).normal(20, 900, n_tr))
    assert monte_carlo_paths(pnls, eq, iters, seed) == _mc_referans(pnls, eq, iters, seed)


def test_mc_paths_bos_ve_iflas():
    """Kenar dallar vektörel gövdede de aynı: boş liste {} döner, iflas yolu ruin sayar."""
    assert monte_carlo_paths([], 100.0, 50) == {} == _mc_referans([], 100.0, 50)
    pnls = [-100_000.0] * 3
    got = monte_carlo_paths(pnls, 1_000.0, 40, 3)
    assert got == _mc_referans(pnls, 1_000.0, 40, 3)
    assert got["ruin_probability"] == 1.0, "her yol iflasla bitmeli"


def test_mc_paths_shuffle_akisi_korunuyor():
    """RNG akışı `random.Random`ta KALMALI — numpy'a taşımak aynı seed'de farklı sayı verir.

    Kanıt doğrudan: aynı seed'le kurulan `random.Random` ile üretilen ilk karıştırma,
    fonksiyonun ürettiği yolların ilkiyle aynı toplamı vermeli."""
    pnls = [10.0, -5.0, 3.0, -8.0, 21.0]
    r = random.Random(42)
    ilk = pnls.copy()
    r.shuffle(ilk)
    curve = [1000.0]
    for p in ilk:
        curve.append(curve[-1] + p)
    # Tüm yollar aynı PnL kümesini taşıdığı için final HEP aynıdır; asıl imza MDD'dir.
    out = monte_carlo_paths(pnls, 1000.0, 1, 42)
    assert out["max_drawdown_p50"] == pytest.approx(max_drawdown(curve), abs=0.0)


# --- 2. mc_bootstrap_terminal BİT-DENK ---------------------------------------------
@pytest.mark.parametrize(("n_ret", "h", "npaths", "blk"), [
    (1500, 21, 800, 5),
    (250, 63, 500, 5),
    (60, 21, 400, 5),
    (10, 21, 200, 5),
    (4, 10, 150, 5),      # n < blk: max_start=1, dilim `arr` boyunda KIRPILIR
    (1500, 5, 600, 7),    # h < blk: tek blok, kuyruk h'ye kesilir
])
def test_bootstrap_bit_denk(n_ret, h, npaths, blk):
    rets = np.random.default_rng(3).normal(0.0005, 0.012, n_ret)
    assert qm.mc_bootstrap_terminal(rets, h, npaths, 7, blk) == \
        _boot_referans(rets, h, npaths, 7, blk)


def test_bootstrap_gecersiz_girdi_hala_sozluk_doner():
    """Sözleşme: İSTİSNA YOK, dönüş TİPİ her zaman dict (vektörleştirme bunu bozmadı)."""
    bos = {"p5": None, "p50": None, "p95": None, "prob_loss": None}
    assert qm.mc_bootstrap_terminal([0.1], 5, 10, 1) == bos      # min_len=2 altı
    assert qm.mc_bootstrap_terminal([0.1, 0.2], 0, 10, 1) == bos  # horizon<=0
    assert qm.mc_bootstrap_terminal([0.1, 0.2], 5, 0, 1) == bos   # n_paths<=0


def test_bootstrap_seed_determinizmi():
    """Aynı seed → aynı çıktı; farklı seed → farklı çıktı (önbellek/sabit dönmüyor)."""
    rets = np.random.default_rng(9).normal(0.0004, 0.01, 500)
    a = qm.mc_bootstrap_terminal(rets, 21, 500, 123)
    assert a == qm.mc_bootstrap_terminal(rets, 21, 500, 123)
    assert a != qm.mc_bootstrap_terminal(rets, 21, 500, 124)


def test_toplu_rng_cekimi_skaler_akisla_ozdes():
    """Vektörleştirmenin DAYANDIĞI varsayım — numpy sürümü değişirse burada kırılsın.

    `mc_bootstrap_terminal` blok başlangıçlarını tek `size=` çağrısıyla çeker; bu ancak
    akış skaler çağrılarla özdeşse bit-denkliği korur."""
    for m in (3, 7, 100, 1496):
        tek = [int(np.random.default_rng(5).integers(0, m))]  # ilk çekim
        akis = np.random.default_rng(5)
        seri = [int(akis.integers(0, m)) for _ in range(64)]
        assert seri[0] == tek[0]
        assert seri == np.random.default_rng(5).integers(0, m, size=64).tolist()


# --- 3. Reddedilen bağımlılıklar ÜRETİMDE OLMAMALI ---------------------------------
def test_reddedilen_bagimliliklar_kurulu_degil():
    """Tur-66/67'de ÖLÇÜLÜP reddedilen paketler `.venv`de bırakılmadı.

    Ölçmek için kurmak meşrudur; ölçtükten sonra bırakmak değildir — kullanılmayan bir
    bağımlılık, Tur-65'in pyarrow dersinin (beyan var / kurulum yok) tersidir."""
    import importlib.util
    for ad, neden in (("winloop", "Tur-66: gerçek hub şeklinde 0,92–0,99× = daha yavaş"),
                      ("numba", "Tur-67: import 969,9 ms + JIT 1211,5 ms; 613 çağrıda başa baş")):
        assert importlib.util.find_spec(ad) is None, f"{ad} kurulu kalmış — {neden}"
