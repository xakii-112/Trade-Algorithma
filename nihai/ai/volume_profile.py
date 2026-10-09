"""
volume_profile.py — HACİM/PİYASA PROFİLİ ve BAR GEOMETRİSİ (Tur-75). KARAR-DIŞI.

`microstructure.py` hacmi ALICI/SATICI olarak ayrıştırır (BVC/VPIN) ve likidite
maliyetini ölçer (Roll/Corwin-Schultz/Amihud/Kyle λ); ikisinde de olmayan aile, hacmin
FİYAT EKSENİNDEKİ dağılımıdır. Dış geliştirme listesinin BÖLÜM 4'ü bunu dokuz maddede
istiyor ve dokuzu da bu deponun ELİNDEKİ veriyle (OHLCV barları) hesaplanabilir —
emir defteri (L2/L3) gerektiren maddeler burada YOKTUR ve gerekçeleri raporda yazılıdır.

| Madde | Fonksiyon |
|---|---|
| 069 POC / Değer Alanı | `volume_profile` |
| 070 Kümülatif Hacim Deltası | `cumulative_volume_delta` |
| 074 Likidite süpürme (sahte kırılım) | `liquidity_sweep` |
| 078 Çıpalanmış VWAP | `anchored_vwap` |
| 080 TPO profili / tek baskılar | `tpo_profile` |
| 083 Bar-içi çarpıklık/basıklık | `intrabar_moments` |
| 084 VWAP σ bantları | `anchored_vwap` (`bands`) |
| 085 Emilim ↔ tükenme | `absorption_score` |
| 105 Adil Değer Boşluğu (FVG) | `fair_value_gaps` |

ANAYASA: KARAR-DIŞI. Hiçbiri sinyale/eşiğe/META_GATE'e bağlanmaz. Listenin her maddesi
bir KURAL öneriyor ("POC'ye dönüşte tepki alımı"); o kurallar BURAYA YAZILMAZ — bir
kenar iddiası `research/edge_lab` kanıt kapısından geçmeden koda giremez (ACI DERS #3/#5).

Sözleşme `quant_math`/`microstructure` ile aynı: kısa/NaN/dönüştürülemez girdi → None,
istisna YOK. Saf fonksiyonlar, numpy dışında bağımlılık yok.

## Üç ölçüm kararı (varsayım gizlenmiyor)

1. **Değer alanı genişletmesi BERABERLİKTE deterministiktir.** Klasik yordam POC'den
   başlayıp her adımda "üst komşu ↔ alt komşu"nun BÜYÜK olanını alır. Eşitlikte bir
   yön seçilmezse aynı veri iki farklı VA verir; burada eşitlik **AŞAĞI** yönü seçer ve
   bu bir konvansiyondur, bir ölçüm değil (`_genislet`).
2. **`fair_value_gaps` boşluğu ÜÇ bar üzerinden tanımlar** (bar i−2'nin ekstremumu ile
   bar i'nin ekstremumu arasında). Bar i−1'in gövdesi KONTROL EDİLMEZ — literatürdeki
   ICT tanımı da etmez; "orta bar büyük olmalı" ek bir filtredir ve `min_size_atr` ile
   çağıranın elindedir.
3. **`absorption_score` bir SKORDUR, olasılık DEĞİLDİR.** Hacim z-skoru ile bar boyu
   z-skorunun FARKIDIR; işareti yorumlanır, büyüklüğü kalibre EDİLMEMİŞTİR (ACI DERS #9'un
   "bir skora Brier uygulanmaz" kuralı — Tur-47).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from nihai.ai.microstructure import bulk_volume_classification

# Değer alanı payı — Market Profile literatürünün standardı (1 standart sapma vekili).
VALUE_AREA_PCT = 0.70
# TPO harf dilimi: 30 dakikalık bar sayısı çağırandan gelir; varsayılan 1 bar = 1 dilim.
DEFAULT_BINS = 50


def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < min_len or not np.all(np.isfinite(arr)):
        return None
    return arr


def _hlv(high: ArrayLike, low: ArrayLike, volume: ArrayLike,
         min_len: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    h = _clean_1d(high, min_len)
    lo = _clean_1d(low, min_len)
    v = _clean_1d(volume, min_len)
    if h is None or lo is None or v is None:
        return None
    if not (h.size == lo.size == v.size) or np.any(v < 0) or np.any(h < lo):
        return None
    return h, lo, v


def _bin_kenarlari(lo: float, hi: float, bins: int) -> np.ndarray | None:
    """[lo, hi] aralığını `bins` eşit kutuya böl. Dejenere aralıkta (hi==lo) None."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or bins < 2:
        return None
    return np.linspace(lo, hi, bins + 1)


# ---------------------------------------------------------------------------
# Madde 069 — Hacim profili: POC ve Değer Alanı
# ---------------------------------------------------------------------------
def _genislet(hist: np.ndarray, poc: int, hedef: float) -> tuple[int, int]:
    """POC'den iki yana genişleyerek kümülatif hacmin `hedef`ini kapsayan [alt, üst].

    Beraberlikte AŞAĞI genişler (deterministik konvansiyon; modül docstring'i §1).
    """
    alt = ust = poc
    toplam = float(hist[poc])
    n = hist.size
    while toplam < hedef and (alt > 0 or ust < n - 1):
        asagi = float(hist[alt - 1]) if alt > 0 else -1.0
        yukari = float(hist[ust + 1]) if ust < n - 1 else -1.0
        if asagi >= yukari:          # eşitlik → AŞAĞI
            alt -= 1
            toplam += asagi
        else:
            ust += 1
            toplam += yukari
    return alt, ust


def volume_profile(high: ArrayLike, low: ArrayLike, volume: ArrayLike,
                   bins: int = DEFAULT_BINS,
                   value_area: float = VALUE_AREA_PCT) -> dict | None:
    """Fiyat ekseninde hacim histogramı → POC · VAH · VAL (Madde 069).

    Her barın hacmi, barın [low, high] aralığını KESEN kutulara **kesişim uzunluğuyla
    orantılı** dağıtılır. Yaygın kısayol hacmin tamamını kapanış kutusuna yazar; o
    yordam geniş barlarda profili yapay olarak sivriltir ve POC'yi kapanışa çeker.

    Dönen: `poc` (en çok hacmin geçtiği fiyat) · `vah`/`val` (değer alanı sınırları) ·
    `hist` (kutu hacimleri) · `edges` (kutu kenarları) · `value_area_share` (GERÇEKLEŞEN
    pay — genişletme kutu granülaritesinde durduğu için hedefi AŞABİLİR).
    Sıfır toplam hacim / dejenere fiyat aralığı / <2 bar → None.
    """
    # Profil TEK bar için de tanımlıdır (o barın aralığına yayılmış hacim).
    hlv = _hlv(high, low, volume, min_len=1)
    if hlv is None:
        return None
    h, lo, v = hlv
    if not (0.0 < value_area <= 1.0):
        return None
    try:
        nb = int(bins)
    except (TypeError, ValueError):
        return None
    edges = _bin_kenarlari(float(lo.min()), float(h.max()), nb)
    if edges is None or float(v.sum()) <= 0.0:
        return None

    # Kesişim uzunluğu matrisi: (bar, kutu). Bar aralığı sıfırsa (h==l) hacim,
    # o fiyatı içeren TEK kutuya gider (0/0 belirsizliğini bölmeye sokmadan).
    alt_k = edges[:-1][None, :]
    ust_k = edges[1:][None, :]
    kesisim = np.clip(np.minimum(h[:, None], ust_k) - np.maximum(lo[:, None], alt_k), 0.0, None)
    genislik = kesisim.sum(axis=1)
    hist = np.zeros(nb, dtype=float)
    genis = genislik > 0
    if np.any(genis):
        hist += (kesisim[genis] / genislik[genis, None] * v[genis, None]).sum(axis=0)
    if np.any(~genis):
        idx = np.clip(np.searchsorted(edges, h[~genis], side="right") - 1, 0, nb - 1)
        np.add.at(hist, idx, v[~genis])

    toplam = float(hist.sum())
    if toplam <= 0.0:
        return None
    poc = int(np.argmax(hist))
    alt, ust = _genislet(hist, poc, value_area * toplam)
    orta = (edges[:-1] + edges[1:]) / 2.0
    return {
        "poc": float(orta[poc]),
        "poc_bin": poc,
        "val": float(edges[alt]),
        "vah": float(edges[ust + 1]),
        "value_area_share": float(hist[alt:ust + 1].sum() / toplam),
        "hist": hist,
        "edges": edges,
        "total_volume": toplam,
        "n_bins": nb,
    }


# ---------------------------------------------------------------------------
# Madde 070 — Kümülatif Hacim Deltası
# ---------------------------------------------------------------------------
def cumulative_volume_delta(prices: ArrayLike, volumes: ArrayLike) -> dict | None:
    """CVD = Σ(V_alım − V_satım), sınıflandırma BVC ile (Madde 070).

    Tick yönü verisi olmadığı için alıcı/satıcı ayrımı `microstructure.
    bulk_volume_classification`tan gelir — YENİ bir sınıflandırıcı yazılmadı (aynı
    varsayımın iki uygulaması sessizce ayrışırdı).

    `divergence`: fiyat pencerede YENİ ZİRVE yaparken CVD yapamıyorsa −1, fiyat yeni
    DİP yaparken CVD yapamıyorsa +1, aksi hâlde 0. Üç durumlu — eşitlik NÖTRDÜR
    (`strategy.vote` disiplini; ACI: eşitliği yön saymak Tur-33/61'de beş kez kusur çıktı).
    CVD'nin BAŞLANGIÇ NOKTASI keyfîdir (ilk bar 0 kabul edilir) → SEVİYESİ değil
    YÖNÜ/eğimi okunur.
    """
    bvc = bulk_volume_classification(prices, volumes)
    if bvc is None:
        return None
    signed = np.asarray(bvc["signed"], dtype=float)
    cvd = np.cumsum(signed)
    p = np.asarray(prices, dtype=float)[1:]        # BVC ilk barı tüketir
    if cvd.size < 2:
        return None
    son_p, son_c = float(p[-1]), float(cvd[-1])
    zirve_p = float(p[:-1].max())
    dip_p = float(p[:-1].min())
    zirve_c = float(cvd[:-1].max())
    dip_c = float(cvd[:-1].min())
    if son_p > zirve_p and son_c <= zirve_c:
        div = -1
    elif son_p < dip_p and son_c >= dip_c:
        div = 1
    else:
        div = 0
    return {
        "cvd": cvd,
        "signed": signed,
        "last": son_c,
        "slope": float(cvd[-1] - cvd[0]) / float(cvd.size - 1),
        "divergence": div,
        "n": int(cvd.size),
    }


# ---------------------------------------------------------------------------
# Madde 078 / 084 — Çıpalanmış VWAP ve σ bantları
# ---------------------------------------------------------------------------
def anchored_vwap(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                  volume: ArrayLike, anchor: int = 0,
                  bands: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict | None:
    """`anchor` barından itibaren kümülatif VWAP + hacim-ağırlıklı σ bantları (078/084).

    Tipik fiyat (H+L+C)/3 kullanılır. Bant yarıçapı, VWAP ETRAFINDAKİ hacim-ağırlıklı
    varyansın kareköküdür — bar getirilerinin std'si DEĞİL. İkisi farklı büyüklüklerdir
    ve karıştırmak bandı sistematik olarak dar gösterir.

    `anchor` negatifse sondan sayılır (`-30` = son 30 bar). Çıpadan sonra <2 bar veya
    çıpa sonrası toplam hacim 0 → None.
    """
    hlv = _hlv(high, low, volume, min_len=1)
    c = _clean_1d(close, 1)
    if hlv is None or c is None:
        return None
    h, lo, v = hlv
    if c.size != h.size:
        return None
    n = h.size
    try:
        a = int(anchor)
    except (TypeError, ValueError):
        return None
    if a < 0:
        a += n
    if a < 0 or a > n - 2:
        return None
    tp = (h[a:] + lo[a:] + c[a:]) / 3.0
    vv = v[a:]
    cum_v = np.cumsum(vv)
    if cum_v[-1] <= 0.0:
        return None
    # cum_v sıfırken VWAP TANIMSIZDIR (0/0); NaN bırakılır, 0 uydurulmaz.
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.cumsum(tp * vv) / cum_v
        # Hacim-ağırlıklı varyans: E[p²] − (E[p])², kümülatif.
        var = np.cumsum(tp * tp * vv) / cum_v - vwap * vwap
    var = np.where(np.isfinite(var), np.clip(var, 0.0, None), np.nan)
    sigma = np.sqrt(var)
    out: dict = {
        "vwap": vwap,
        "sigma": sigma,
        "anchor": a,
        "n": int(vwap.size),
        "last_vwap": float(vwap[-1]) if np.isfinite(vwap[-1]) else None,
        "last_z": None,
        "bands": {},
    }
    for k in bands:
        try:
            kf = float(k)
        except (TypeError, ValueError):
            continue
        out["bands"][kf] = {"upper": vwap + kf * sigma, "lower": vwap - kf * sigma}
    if np.isfinite(sigma[-1]) and sigma[-1] > 0.0:
        out["last_z"] = float((c[-1] - vwap[-1]) / sigma[-1])
    return out


# ---------------------------------------------------------------------------
# Madde 080 — TPO / Market Profile ve tek baskılar
# ---------------------------------------------------------------------------
def tpo_profile(high: ArrayLike, low: ArrayLike, bars_per_bracket: int = 1,
                bins: int = DEFAULT_BINS) -> dict | None:
    """Zaman-Fiyat-Fırsatı profili: her fiyat kutusuna DEĞEN dilim sayısı (Madde 080).

    TPO hacimden bağımsızdır — fiyatın bir seviyede ne kadar ZAMAN geçirdiğini sayar.
    `single_prints`, YALNIZ BİR dilimin değdiği kutulardır (hızlı geçiş = likidite
    boşluğu). Sayım kutu başına DİLİM sayısıdır, bar sayısı değil; bu yüzden
    `bars_per_bracket` (30 dk için: 30/bar_dakikası) çağırandan gelir ve
    varsayılan 1 "her bar bir dilim" demektir.

    `single_prints` KENARLARDA da doğar (ilk/son dilimin uçları); bunlar gerçek boşluk
    değil profilin SINIRIDIR ve `single_prints_interior` onları ayıklar.
    """
    hlv = _hlv(high, low, np.ones(np.asarray(high).size))
    if hlv is None:
        return None
    h, lo, _ = hlv
    try:
        bpb = max(1, int(bars_per_bracket))
        nb = int(bins)
    except (TypeError, ValueError):
        return None
    edges = _bin_kenarlari(float(lo.min()), float(h.max()), nb)
    if edges is None:
        return None
    n = h.size
    n_dilim = int(np.ceil(n / bpb))
    sayim = np.zeros(nb, dtype=int)
    for d in range(n_dilim):
        seg = slice(d * bpb, min((d + 1) * bpb, n))
        d_lo, d_hi = float(lo[seg].min()), float(h[seg].max())
        # Kutuya DEĞME testi: kutu ile dilim aralığı kesişiyor mu (kapalı aralık).
        deger = (edges[1:] > d_lo) & (edges[:-1] < d_hi)
        if not np.any(deger):        # dejenere dilim (d_lo == d_hi) → içeren kutu
            idx = min(int(np.searchsorted(edges, d_hi, side="right") - 1), nb - 1)
            deger = np.zeros(nb, dtype=bool)
            deger[max(idx, 0)] = True
        sayim += deger.astype(int)
    dolu = np.flatnonzero(sayim > 0)
    tek = np.flatnonzero(sayim == 1)
    ic_tek = tek[(tek > dolu.min()) & (tek < dolu.max())] if dolu.size else tek
    orta = (edges[:-1] + edges[1:]) / 2.0
    return {
        "tpo_count": sayim,
        "edges": edges,
        "n_brackets": n_dilim,
        "poc_tpo": float(orta[int(np.argmax(sayim))]),
        "single_prints": [float(x) for x in orta[tek]],
        "single_prints_interior": [float(x) for x in orta[ic_tek]],
    }


# ---------------------------------------------------------------------------
# Madde 085 — Emilim ↔ tükenme
# ---------------------------------------------------------------------------
def absorption_score(high: ArrayLike, low: ArrayLike, volume: ArrayLike,
                     window: int = 20) -> dict | None:
    """Hacim z-skoru − bar boyu z-skoru (Madde 085). SKORDUR, olasılık DEĞİLDİR.

    POZİTİF = yüksek hacim / dar bar → arz-talep EMİLİYOR (kurumsal emir tahtayı
    tutuyor). NEGATİF = düşük hacim / geniş bar → hareket TÜKENİYOR (likidite yok).

    z-skorları YÜRÜYENDİR ve pencere dolmadan üretilmez; her iki bileşenin standart
    sapması sıfırsa (düz seri) o bar için skor NaN'dır — sıfır "nötr emilim" değil
    ÖLÇÜM YOKLUĞUDUR (Tur-31/33 kuralı).
    """
    hlv = _hlv(high, low, volume)
    if hlv is None:
        return None
    h, lo, v = hlv
    try:
        w = int(window)
    except (TypeError, ValueError):
        return None
    if w < 3 or h.size < w:
        return None
    rng = h - lo

    def _z(x: np.ndarray) -> np.ndarray:
        # sliding_window_view ile yürüyen ortalama/std (pencere DOLMADAN NaN).
        win = np.lib.stride_tricks.sliding_window_view(x, w)
        mu = win.mean(axis=1)
        sd = win.std(axis=1, ddof=1)
        z = np.full(x.size, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            z[w - 1:] = np.where(sd > 0, (x[w - 1:] - mu) / sd, np.nan)
        return z

    zv, zr = _z(v), _z(rng)
    skor = zv - zr
    gecerli = np.isfinite(skor)
    return {
        "score": skor,
        "vol_z": zv,
        "range_z": zr,
        "last": float(skor[-1]) if gecerli[-1] else None,
        "n_measured": int(gecerli.sum()),
    }


# ---------------------------------------------------------------------------
# Madde 074 — Likidite süpürme (sahte kırılım)
# ---------------------------------------------------------------------------
def liquidity_sweep(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                    lookback: int = 20) -> dict | None:
    """Önceki `lookback` barın ekstremumunu DELİP İÇİNE KAPANAN barlar (Madde 074).

    `sweep_high[i] = 1` ⇔ high[i] > max(high[i−lookback : i]) VE close[i] < o eşik.
    Yani seviye gün içinde alınmış ama bar KAPANIŞTA geri dönmüştür — kırılım
    doğrulanmamış, stop havuzu süpürülmüştür.

    Ekstremum penceresi barın KENDİSİNİ İÇERMEZ (`[i−lookback : i]`); içerseydi koşul
    yapısı gereği hiç sağlanmazdı. İlk `lookback` bar için değer 0'dır — ölçüm yokluğu
    burada NaN yerine 0'dır çünkü çıktı bir OLAY BAYRAĞIDIR ve "olay olmadı" doğru
    cevaptır (`absorption_score`taki skorla karıştırılmamalı).
    """
    hlv = _hlv(high, low, np.ones(np.asarray(high).size))
    c = _clean_1d(close, 2)
    if hlv is None or c is None:
        return None
    h, lo, _ = hlv
    if c.size != h.size:
        return None
    try:
        lb = int(lookback)
    except (TypeError, ValueError):
        return None
    if lb < 1 or h.size <= lb:
        return None
    n = h.size
    up = np.zeros(n, dtype=int)
    dn = np.zeros(n, dtype=int)
    # Kayan pencere ekstremumları — pencere bar i'nin GERİSİNDE biter.
    hw = np.lib.stride_tricks.sliding_window_view(h[:-1], lb).max(axis=1)
    lw = np.lib.stride_tricks.sliding_window_view(lo[:-1], lb).min(axis=1)
    idx = np.arange(lb, n)
    up[idx] = ((h[idx] > hw) & (c[idx] < hw)).astype(int)
    dn[idx] = ((lo[idx] < lw) & (c[idx] > lw)).astype(int)
    return {
        "sweep_high": up,
        "sweep_low": dn,
        "n_sweep_high": int(up.sum()),
        "n_sweep_low": int(dn.sum()),
        "last": int(up[-1] - dn[-1]),      # +1 üst süpürme, −1 alt, 0 yok
        "lookback": lb,
    }


# ---------------------------------------------------------------------------
# Madde 105 — Adil Değer Boşluğu (FVG)
# ---------------------------------------------------------------------------
def fair_value_gaps(high: ArrayLike, low: ArrayLike,
                    min_size: float = 0.0) -> dict | None:
    """Üç barlık dengesizlik boşlukları (Madde 105).

    Yükseliş FVG: `low[i] > high[i−2]` → [high[i−2], low[i]] aralığı hiç işlem görmedi.
    Düşüş FVG: `high[i] < low[i−2]` → [high[i], low[i−2]].
    `min_size` mutlak fiyat biriminde alt sınırdır (0 = filtre yok).

    `filled`: boşluk DOĞDUKTAN SONRAKİ barlarda fiyatın aralığa GİRİP girmediği. Ölçüm
    NEDENSELDİR — yalnız i'den sonraki barlara bakılır; boşluğun kendi barı doldurma
    sayılmaz (aksi hâlde tanım gereği her boşluk kendi barında dolardı).
    """
    hlv = _hlv(high, low, np.ones(np.asarray(high).size), min_len=3)
    if hlv is None:
        return None
    h, lo, _ = hlv
    try:
        ms = float(min_size)
    except (TypeError, ValueError):
        return None
    if ms < 0.0:
        return None
    n = h.size
    kayitlar: list[dict] = []
    for i in range(2, n):
        for yon, alt, ust in ((1, h[i - 2], lo[i]), (-1, h[i], lo[i - 2])):
            boy = float(ust - alt)
            if boy <= 0.0 or boy < ms:
                continue
            sonrasi = slice(i + 1, n)
            doldu = bool(np.any((h[sonrasi] >= alt) & (lo[sonrasi] <= ust)))
            kayitlar.append({
                "index": int(i), "direction": int(yon),
                "lower": float(alt), "upper": float(ust), "size": boy,
                "filled": doldu,
            })
    acik = [g for g in kayitlar if not g["filled"]]
    return {
        "gaps": kayitlar,
        "n_gaps": len(kayitlar),
        "n_open": len(acik),
        "open_gaps": acik,
        "fill_rate": (float(len(kayitlar) - len(acik)) / len(kayitlar)) if kayitlar else None,
    }


# ---------------------------------------------------------------------------
# Madde 083 — Bar-içi çarpıklık ve basıklık
# ---------------------------------------------------------------------------
def intrabar_moments(returns: ArrayLike) -> dict | None:
    """Bar-içi getiri serisinin 3. ve 4. momentleri (Madde 083).

    Girdi, BİR barı oluşturan alt-bar (ör. 1m) LOG getirileridir; tick verisi yoksa
    doğru vekil budur. `excess_kurtosis` normal dağılıma göredir (Fisher tanımı: normal
    için 0). `skew`/`kurtosis` örneklem momentlerinden hesaplanır ve **yansızlaştırma
    UYGULANMAZ** — küçük n'de yanlıdır ve `n` çıktıda taşınır (Tur-34: ölçüm KOŞULU
    ölçümün kendisi kadar taşınmalıdır).

    <4 gözlem veya σ=0 (düz seri) → None; düz seride çarpıklık 0/0'dır ve 0 basmak
    "simetrik dağılım" demek olurdu (Tur-33 dejenere-girdi dersi).
    """
    r = _clean_1d(returns, 4)
    if r is None:
        return None
    mu = float(r.mean())
    d = r - mu
    m2 = float(np.mean(d ** 2))
    if m2 <= 0.0:
        return None
    m3 = float(np.mean(d ** 3))
    m4 = float(np.mean(d ** 4))
    sd = np.sqrt(m2)
    return {
        "mean": mu,
        "std": float(sd),
        "skew": m3 / (sd ** 3),
        "kurtosis": m4 / (m2 * m2),
        "excess_kurtosis": m4 / (m2 * m2) - 3.0,
        "n": int(r.size),
    }
