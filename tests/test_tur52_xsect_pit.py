"""
test_tur52_xsect_pit.py — Tur-52 kilitleri (kesitsel momentum, PIT ölçümü).

İki kusur sınıfını çapalar; ikisi de ölçülerek bulundu:

* **BULGU-52-2** — al-tut kontrolü uç SATIRI okuyordu (`panel.iloc[first/t]`). Bu yalnız
  tüm sembollerin aynı gün başlayıp bittiği panelde doğrudur. PIT panelinde isimler ÖLÜR
  ve yerel dosyalar farklı günlerde biter: son satırda 525 sembolün 4'ünde fiyat vardı →
  `bh_rets` boş → kontrol sessizce **0.0** döndü, yani §6'nın beta tuzağını yakalayacak
  kontrolün KENDİSİ kayboldu ve momentum "beta'yı aştı" gibi göründü.
* **BULGU-52-3** — `edge_vs_buyhold` TOPLAM getiri kıyaslar. 10 isimlik yoğun bir portföy
  525 isimlik sepetten zaten daha oynaktır; toplam getiride öne geçmek RİSK BİRİMİ BAŞINA
  öne geçmek değildir. `buyhold_sharpe` bu ikisini ayırmak için vardır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.xsect import run_momentum

_KW = dict(lookback=20, skip=2, hold=5, top=3, cost_pct=0.0)


def _panel(n_bar: int = 200, n_sym: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n_bar)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (n_bar, n_sym)), axis=0))
    return pd.DataFrame(px, index=idx, columns=[f"S{i}" for i in range(n_sym)])


def test_dolu_panelde_al_tut_ESKI_davranisla_denk():
    """Düzeltme yalnız DÜZENSİZ paneli etkilemeli — dolu panelde ilk/son geçerli = ilk/son satır."""
    panel = _panel()
    m = run_momentum(panel, **_KW)
    first = _KW["lookback"] + 1
    t = first
    while t + _KW["hold"] < len(panel):
        t += _KW["hold"]
    eski = float(np.mean([panel[s].iloc[t] / panel[s].iloc[first] - 1.0 for s in panel.columns])) * 100
    assert m["buyhold_return_pct"] == round(eski, 1)


def test_OLEN_isimli_panelde_kontrol_SIFIRLANMAZ():
    """Kusurun kendisi: ragged panelde al-tut kontrolü kaybolmamalı."""
    panel = _panel()
    panel.iloc[-30:, :7] = np.nan            # 10 sembolün 7'si erken 'ölüyor'
    m = run_momentum(panel, **_KW)
    assert m, "metrik üretilemedi"
    assert m["buyhold_return_pct"] != 0.0, "kontrol sessizce sıfırlandı (BULGU-52-2)"


def test_olen_isim_al_tut_a_KAYBIYLA_katilir():
    """Ölen isim son işlem fiyatıyla sayılmalı — dışlanırsa al-tut yukarı yanlı olur."""
    panel = _panel(n_sym=4, seed=3)
    panel.iloc[100:, 0] = np.nan                       # S0 100. barda kotasyondan çıkıyor
    panel.iloc[95:100, 0] = panel.iloc[95, 0] * 0.1    # ...çökerek
    m = run_momentum(panel, **_KW)
    hepsi_hayatta = run_momentum(_panel(n_sym=4, seed=3), **_KW)
    assert m["buyhold_return_pct"] < hepsi_hayatta["buyhold_return_pct"]


def test_risk_duzeltilmis_kiyas_RAPORLANIR():
    """`buyhold_sharpe` olmadan 'kenar' iddiası yalnız yoğunlaşma etkisi olabilir."""
    m = run_momentum(_panel(), **_KW)
    assert "buyhold_sharpe" in m and isinstance(m["buyhold_sharpe"], float)
