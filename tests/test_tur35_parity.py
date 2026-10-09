"""Tur-35 OPTİMİZASYON PARİTESİ — hızlandırılan üç gövdenin denkliği KİLİTLENİR.

`tests/test_perf_parity_round10.py` deseninin devamı: her testte Tur-34'ün gövdesi
REFERANS olarak elde yeniden yazılır ve yeni gövdeyle aynı girdide karşılaştırılır.
Referansı burada tutmanın nedeni, "hızlandırdım ve galiba aynı" cümlesinin bir sonraki
turda kanıtsız kalmamasıdır — iddia koda gömülür, yoruma değil.

Kapsananlar:
  1. `neural._auc_np`            — beraberlik döngüsü → vektörize (BİT-DENK iddiası)
  2. `online.hedge_combine`      — adım-başı kayıp → önceden hesaplanan matris (BİT-DENK)
  3. `atomic_io.tail_lines`      — kuadratik kuyruk okuma → O(bayt) (ÇIKTI-DENK, n>0)
  4. üç gösterge                 — `pd.concat(...).max(axis=1)` → `np.fmax` (BİT-DENK)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nihai.ai import neural, online
from nihai.atomic_io import tail_lines
from nihai.signals import indicators as I
from nihai.signals import indicators_ext as E


# ---------------------------------------------------------------------------
# Tur-34 REFERANS gövdeleri (elle taşındı — değiştirilmemeli)
# ---------------------------------------------------------------------------
def _auc_ref(y, p):
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    if y_arr.size != p_arr.size or y_arr.size == 0:
        return None
    pos = y_arr >= 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p_arr, kind="mergesort")
    sorted_p = p_arr[order]
    ranks = np.empty(p_arr.size, dtype=float)
    i = 0
    while i < sorted_p.size:
        j = i
        while j + 1 < sorted_p.size and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _hedge_ref(preds, y, eta, share):
    n, k = preds.shape
    w = np.full(k, 1.0 / k, dtype=float)
    combined = np.empty(n, dtype=float)
    weights_hist = np.empty((n, k), dtype=float)
    for t in range(n):
        weights_hist[t] = w
        combined[t] = float(np.dot(w, preds[t]))
        losses = online._log_loss(preds[t], np.full(k, y[t]))
        w = w * np.exp(-eta * (losses - np.min(losses)))
        total = float(np.sum(w))
        w = np.full(k, 1.0 / k) if total <= 0 or not math.isfinite(total) else w / total
        if share > 0:
            w = (1.0 - share) * w + share / k
    return combined, weights_hist, w


def _tail_ref(path, n):
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            data = b""
            while pos > 0 and data.count(b"\n") <= n:
                step = min(8192, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data
    except OSError:
        return []
    return [p.decode("utf-8", "replace") for p in data.split(b"\n") if p.strip()][-n:]


# ---------------------------------------------------------------------------
# 1) _auc_np — BİT-DENK
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 7, 64, 1000])
@pytest.mark.parametrize("kind", ["surekli", "cok_bagli", "hepsi_ayni", "nan", "inf", "negatif"])
def test_auc_np_bit_denk(n, kind):
    """Beraberlik döngüsünün vektörleştirilmesi BİT-DENKTİR — yaklaşık değil, birebir.

    NaN dalı özellikle önemli: özgün döngü `sorted_p[j+1] == sorted_p[i]` diyordu ve
    NaN != NaN olduğu için her NaN KENDİ grubunu kuruyordu. Yeni gövde grup sınırını
    komşu karşılaştırmasıyla bulur; aynı NaN semantiğini korumak zorundadır."""
    rs = np.random.RandomState(1000 + n + hash(kind) % 997)
    p = rs.rand(n)
    if kind == "cok_bagli":
        p = np.round(p, 1)
    elif kind == "hepsi_ayni":
        p = np.full(n, 0.25)
    elif kind == "negatif":
        p = p * 10.0 - 5.0
    elif kind == "nan" and n >= 4:
        p[rs.rand(n) < 0.25] = np.nan
    elif kind == "inf" and n >= 4:
        p[0], p[1] = np.inf, -np.inf
    y = (rs.rand(n) > 0.5).astype(float)
    ref, new = _auc_ref(y, p), neural._auc_np(y, p)
    if ref is None or new is None:
        assert ref is None and new is None
        return
    # Bit-denklik: float64 gösterimi BİREBİR aynı olmalı (np.isclose YETMEZ).
    assert np.float64(ref).tobytes() == np.float64(new).tobytes()


def test_auc_np_bilinen_deger():
    """Kapalı-form kontrol: kusursuz ayrım → 1.0, ters ayrım → 0.0, tam beraberlik → 0.5."""
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert neural._auc_np(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert neural._auc_np(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert neural._auc_np(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_auc_np_tek_sinif_none():
    assert neural._auc_np(np.ones(5), np.random.rand(5)) is None
    assert neural._auc_np(np.zeros(5), np.random.rand(5)) is None
    assert neural._auc_np(np.array([]), np.array([])) is None


# ---------------------------------------------------------------------------
# 2) hedge_combine — BİT-DENK
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("n", "k", "share"), [(20, 2, 0.0), (200, 3, 0.02),
                                               (150, 5, 0.5), (80, 2, 1.0)])
def test_hedge_combine_bit_denk(n, k, share):
    """Kayıp matrisini döngü ÖNCESİ hesaplamak nedenselliği bozmaz (kayıp ağırlığa bağlı
    DEĞİLDİR) ve `_log_loss` eleman-bazlı olduğu için sonuç BİREBİR aynıdır."""
    rs = np.random.RandomState(7 + n)
    preds = np.clip(rs.rand(n, k), 0.001, 0.999)
    y = (rs.rand(n) > 0.5).astype(float)
    c_ref, w_ref, f_ref = _hedge_ref(preds, y, 1.3, share)
    res = online.hedge_combine(preds, y, eta=1.3, share=share, include_base=False)
    assert res is not None
    assert np.array_equal(c_ref, res["combined"])
    assert np.array_equal(w_ref, res["weights"])
    assert np.array_equal(f_ref, res["final_weights"])


def test_hedge_combine_gecersiz_girdi_none():
    """Savunmacı sözleşme korundu: uyumsuz/aralık-dışı girdi → None (çökme yok)."""
    y = np.array([0, 1, 0, 1])
    assert online.hedge_combine(np.random.rand(3, 2), y) is None          # uzunluk
    assert online.hedge_combine(np.full((4, 2), 1.5), y) is None          # aralık dışı
    assert online.hedge_combine(np.random.rand(4, 2), y, eta=0.0) is None  # eta>0 şartı
    assert online.hedge_combine(np.random.rand(4, 2), y, share=1.5) is None


# ---------------------------------------------------------------------------
# 3) tail_lines — ÇIKTI-DENK (n>0) + n<=0 kusuru kapandı
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 37, 500, 100_000])
def test_tail_lines_cikti_denk(tmp_path, n):
    """Kuadratik gövde ile yeni O(bayt) gövde AYNI satırları döndürür. Dosya blok
    sınırlarını kesecek kadar büyük seçilir (16 KB'lik ilk bloktan büyük)."""
    p = tmp_path / "log.csv"
    # write_bytes: Windows'ta write_text "\n" → "\r\n" çevirir ve satır içeriği değişirdi.
    p.write_bytes("".join(f"satir-{i},{i * 3},x{i % 7}\n" for i in range(4000)).encode())
    assert _tail_ref(p, n) == tail_lines(p, n)


@pytest.mark.parametrize("kind", ["son_newline_yok", "tek_satir_newline_yok", "bos_satirlar",
                                  "crlf", "utf8_bozuk", "tek_dev_satir"])
@pytest.mark.parametrize("n", [1, 3, 50, 5000])
def test_tail_lines_dejenere_dosyalar(tmp_path, kind, n):
    """DEJENERE girdi sondajı (Tur-28/33 disiplini). Append-only canlı log eşzamanlı
    YAZILIRKEN okunur: son satırın newline'ı henüz gelmemiş olabilir, satır CRLF'li
    olabilir, UTF-8 bozulabilir. Blok sınırını kesen tek dev satır da ayrıca sınanır."""
    icerik = {
        "son_newline_yok": b"".join(f"s{i}\n".encode() for i in range(3000))[:-1],
        "tek_satir_newline_yok": b"yalniz-bir-satir",
        "bos_satirlar": b"".join((b"\n" if i % 3 else f"s{i}\n".encode()) for i in range(3000)),
        "crlf": b"".join(f"s{i}\r\n".encode() for i in range(3000)),
        "utf8_bozuk": b"".join(f"s{i}".encode() + b"\xff\xfe\n" for i in range(3000)),
        "tek_dev_satir": b"A" * 300_000 + b"\n" + b"son\n",
    }[kind]
    p = tmp_path / "log.csv"
    p.write_bytes(icerik)
    assert _tail_ref(p, n) == tail_lines(p, n)


def test_tail_lines_bos_ve_olmayan_dosya(tmp_path):
    bos = tmp_path / "bos.csv"
    bos.write_text("", encoding="utf-8")
    assert tail_lines(bos, 10) == []
    assert tail_lines(tmp_path / "yok.csv", 10) == []


def test_tail_lines_n_sifir_bos_liste(tmp_path):
    """ESKİ KUSUR: `lst[-0:]` Python'da `lst[0:]`dir — "sıfır satır iste" TÜM okunan bloğu
    döndürüyordu ve satır sayısı BLOK BOYUTUNA bağlıydı (tanımsız davranış).
    `AI_REASONER_PRED_TAIL=0` ile tetiklenebilirdi."""
    p = tmp_path / "log.csv"
    p.write_bytes("".join(f"s{i}\n" for i in range(500)).encode())
    assert tail_lines(p, 0) == []
    assert tail_lines(p, -3) == []


def test_tail_lines_yarim_ilk_satir_dislanir(tmp_path):
    """Blok sınırında kalan YARIM satır çıktıya SIZMAMALI: döngü n+1 satır sonu biriktirir,
    bu yüzden yarım parça her zaman `[-n:]` diliminin dışında kalır."""
    p = tmp_path / "log.csv"
    p.write_bytes("".join(f"{'A' * 200}-{i}\n" for i in range(300)).encode())
    for n in (1, 5, 50, 299):
        out = tail_lines(p, n)
        assert len(out) == n
        assert all(line.startswith("A" * 200) for line in out)
        assert out[-1] == f"{'A' * 200}-299"


def test_tail_lines_tek_kaynak_uc_modulde():
    """Üç modüldeki `_tail_lines` artık AYNI gövdeye ince sarmalayıcıdır (kopya YOK)."""
    from nihai.ai import calibration_loop, learning, reasoner
    for mod in (learning, calibration_loop, reasoner):
        assert mod._tail_lines.__doc__ is not None
        assert "atomic_io.tail_lines" in mod._tail_lines.__doc__


# ---------------------------------------------------------------------------
# 4) Gösterge hattı — pd.concat(...).max(axis=1) → np.fmax (BİT-DENK)
# ---------------------------------------------------------------------------
def _uo_ref(df, s=7, m=14, lng=28):
    prev_close = df["close"].shift()
    bp = df["close"] - pd.concat([df["low"], prev_close], axis=1).min(axis=1)
    tr = (pd.concat([df["high"], prev_close], axis=1).max(axis=1)
          - pd.concat([df["low"], prev_close], axis=1).min(axis=1)).replace(0, np.nan)
    a1 = bp.rolling(s).sum() / tr.rolling(s).sum()
    a2 = bp.rolling(m).sum() / tr.rolling(m).sum()
    a3 = bp.rolling(lng).sum() / tr.rolling(lng).sum()
    return 100 * (4 * a1 + 2 * a2 + a3) / 7.0


def _wad_ref(df):
    c, pc = df["close"], df["close"].shift()
    trh = pd.concat([df["high"], pc], axis=1).max(axis=1)
    trl = pd.concat([df["low"], pc], axis=1).min(axis=1)
    move = np.where(c > pc, c - trl, np.where(c < pc, c - trh, 0.0))
    return pd.Series(move, index=df.index).cumsum()


def _si_ref(df, limit=3.0):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    pc, po = c.shift(), o.shift()
    k = pd.concat([(h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr = pd.concat([(h - pc).abs(), (l - pc).abs(), (h - l).abs()], axis=1).max(axis=1)
    si = 50 * ((c - pc) + 0.5 * (c - o) + 0.25 * (pc - po)) / tr.replace(0, np.nan) * (k / limit)
    return si.fillna(0.0)


def _mk_df(seed: int, n: int, kind: str) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    c = 100 * np.cumprod(1 + rs.randn(n) * 0.01)
    df = pd.DataFrame({"open": c * (1 + rs.randn(n) * 0.001),
                       "high": c * (1 + abs(rs.randn(n)) * 0.004),
                       "low": c * (1 - abs(rs.randn(n)) * 0.004),
                       "close": c,
                       "volume": rs.randint(int(1e5), int(1e6), n).astype(float)})
    if kind == "duz":       # dejenere: tamamen düz pencere (Tur-28/33 dersi)
        df.loc[:, ["open", "high", "low", "close"]] = 100.0
    if kind == "nan":
        df.loc[df.sample(frac=0.05, random_state=1).index, "high"] = np.nan
        df.loc[df.sample(frac=0.05, random_state=2).index, "low"] = np.nan
    return df


@pytest.mark.parametrize("kind", ["normal", "duz", "nan"])
@pytest.mark.parametrize("n", [40, 120, 400])
def test_gosterge_fmax_bit_denk(kind, n):
    """Tur-6 `strategy._true_range`'te bu deseni kaldırmıştı; gösterge modüllerinde DÖRT
    yerde yaşamaya devam ediyordu (ders MODÜLE değil DESENE aitti). `np.fmax/fmin` NaN'ı
    yok sayar = pandas skipna → değerler BİREBİR aynı, NaN deseni de aynı."""
    df = _mk_df(seed=n + len(kind), n=n, kind=kind)
    for ref, new in ((_uo_ref(df), I.ultimate_oscillator(df)),
                     (_wad_ref(df), E.williams_ad(df)),
                     (_si_ref(df), E.swing_index(df))):
        a, b = ref.to_numpy(dtype=float), new.to_numpy(dtype=float)
        assert np.array_equal(np.isnan(a), np.isnan(b))     # NaN deseni birebir
        assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])   # değerler bit-denk
