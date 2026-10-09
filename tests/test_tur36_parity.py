"""Tur-36 optimizasyon PARİTE testleri — hızlandırılan yolların çıktısı korunuyor mu?

İki optimizasyon:
  1. `learning.update_from_predictions` satır okuma (sözlük arama + kapanış çağrısı →
     önceden çözülmüş indeks). İddia: ÇIKTI-DENK — dejenere satırlarda da.
  2. `quant_math`/`options_math` skaler normal dağılım (scipy → stdlib `NormalDist`).
     ÜÇ FARKLI iddia: `pdf` BİT-DENK · `cdf` ≤2,3e-16 · `ppf` ≤1,8e-15 — ve ayrıca
     KARAR düzeyinde (opsiyon fiyatı ≥1 sent → bağıl fark ≤1e-11) kilitlenir.
"""
from __future__ import annotations

import math
import statistics

import numpy as np
import pytest
from scipy import stats

from nihai.ai import learning, options_math, quant_math

# ---------------------------------------------------------------------------
# 1. Satır okuma — dejenere CSV satırları
# ---------------------------------------------------------------------------
HEADER = ("logged_at,symbol,interval,bar_ts,direction,action,confidence,prob_cal,pwin,"
          "pwin_side,price,bar_close,horizon_bars,resolved_at,fwd_price,hit,fwd_ret_pct,"
          "hit_bar,slip_bps,neural_pwin,neural_pwin_side")


def _write_log(tmp_path, monkeypatch, rows: list[str]):
    p = tmp_path / "predictions_log.csv"
    p.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(learning, "PRED_LOG", p)
    monkeypatch.setattr(learning, "LEARNING_STATE", tmp_path / "state.json")
    return p


def test_dejenere_satirlar_sessizce_bozmaz(tmp_path, monkeypatch):
    """KISA satır (eksik hücre), UZUN satır (fazla hücre), BOŞLUKLU değerler ve tamamen
    boş satır — hiçbiri çökmemeli ve YALNIZ geçerli satırlar sayılmalı.

    Eski `cell()` kapanışı bu üç durumu sınır kontrolüyle çözüyordu; yeni indeksleme
    satırı `n_col + 1` hücreye normalize ederek çözer — sözleşme AYNI kalmalı."""
    tam = ("2026-01-01T10:00:00,AAA,1m,2026-01-01 10:00:00,+,BUY, 0.60 , 0.55 , 0.70 ,"
           "LONG,10,10,5,2026-01-01T11:00:00,11,1,1.5,3,2,0.65,LONG")
    kisa = "2026-01-01T10:00:00,BBB,1m,2026-01-01 10:00:00,+"
    uzun = tam + ",FAZLA,HUCRE,VAR"
    bos = ""
    _write_log(tmp_path, monkeypatch, [tam, kisa, uzun, bos])
    st = learning.update_from_predictions(tail_n=100)
    # `tam` + `uzun` çözülmüş ve hit'li → 2 satır; `kisa`/`bos` resolved_at taşımaz.
    assert st["strategies"]["GLOBAL"]["n"] == 2
    assert st["strategies"]["LONG"]["n"] == 2
    assert st["n_unparsed"] == 0


def test_bosluklu_degerler_kirpilmis_gibi_okunur(tmp_path, monkeypatch):
    """`_f`/`_norm_dir` girdiyi kendileri kırpar → `.strip()` kaldırılması ÇIKTIYI
    değiştirmemeli (optimizasyonun dayandığı denklik iddiası budur)."""
    with_ws = ("2026-01-01T10:00:00,AAA,1m, 2026-01-01 10:00:00 , + ,BUY,0.6,0.55, 0.70 ,"
               " LONG ,10,10,5,2026-01-01T11:00:00,11, 1 , 1.5 ,3,2, 0.65 , LONG ")
    _write_log(tmp_path, monkeypatch, [with_ws])
    st = learning.update_from_predictions(tail_n=100)
    assert st["strategies"]["LONG"]["n"] == 1
    assert st["n_unparsed"] == 0                     # bar_ts boşluklu ama ayrıştırılabilir
    assert st["sources"]["pwin"]["n"] == 1


def test_olmayan_kolon_bos_hucre_verir(tmp_path, monkeypatch):
    """Şemada OLMAYAN kolon (`neural_pwin`) — eski kod "" döndürüyordu; yeni kod satırın
    sonundaki daima-boş hücreyi gösteriyor. Sonuç: `sources.neural` doldurulmaz."""
    kisa_header = "logged_at,symbol,bar_ts,direction,pwin,pwin_side,resolved_at,hit,fwd_ret_pct"
    p = tmp_path / "predictions_log.csv"
    p.write_text(kisa_header + "\n"
                 "2026-01-01T10:00:00,AAA,2026-01-01 10:00:00,+,0.7,LONG,"
                 "2026-01-01T11:00:00,1,1.5\n", encoding="utf-8")
    monkeypatch.setattr(learning, "PRED_LOG", p)
    monkeypatch.setattr(learning, "LEARNING_STATE", tmp_path / "state.json")
    st = learning.update_from_predictions(tail_n=100)
    assert st["strategies"]["GLOBAL"]["n"] == 1
    assert st["sources"]["neural"]["n"] == 0
    assert st["source_fusion"] is None               # neural olasılığı yok → hizalı üçlü yok


# ---------------------------------------------------------------------------
# 2. Skaler normal dağılım — scipy ↔ stdlib
# ---------------------------------------------------------------------------
def test_norm_pdf_scipy_ile_bit_denk():
    """`pdf` ÖLÇÜLDÜ: 20.002 noktanın %100'ünde fark TAM SIFIR → iddia BİT-DENKtir.
    Diğer ikisi için aynı iddia KURULAMAZ (aşağıdaki test) — hangi iddianın
    kurulabileceğini algoritma belirler, umut değil (Tur-34 kuralı)."""
    rs = np.random.RandomState(0)
    for x in np.concatenate([rs.randn(500) * 3.0, [0.0, 8.0, -8.0]]):
        assert quant_math._norm_pdf(float(x)) == float(stats.norm.pdf(x))


def test_norm_cdf_ppf_olculmus_sinir_icinde():
    """`cdf`/`ppf` bit-denk DEĞİLDİR (iki kütüphane aynı matematiği farklı sırayla yapar).
    İddia ölçülmüş bir SINIRDIR: cdf ≤ 2,3e-16 · ppf ≤ 1,8e-15 mutlak."""
    rs = np.random.RandomState(0)
    for x in np.concatenate([rs.randn(2000) * 3.0, [0.0, 8.0, -8.0, 40.0, -40.0]]):
        assert abs(quant_math._norm_cdf(float(x)) - float(stats.norm.cdf(x))) <= 2.3e-16
    for p in np.concatenate([rs.rand(2000), [1e-6, 0.025, 0.5, 0.975, 1 - 1e-6]]):
        assert abs(quant_math._norm_ppf(float(p)) - float(stats.norm.ppf(p))) <= 1.8e-15


def test_bs_price_karar_duzeyinde_denk():
    """KARAR düzeyi: 1 sentin üstündeki her fiyatta bağıl fark ≤ 1e-11 ÖLÇÜLDÜ.
    (Bağıl fark yalnız fiyat ~0'a giderken büyür; orada mutlak fark 1e-13 mertebesindedir
    ve "değersiz opsiyon" bölgesidir — dürüstlük kalemi, bkz. Tur-36 raporu §7.)"""
    nd = statistics.NormalDist()
    assert nd is not None
    rs = np.random.RandomState(1)
    n_kontrol = 0
    for _ in range(400):
        s = float(rs.uniform(5, 500))
        k = s * float(rs.uniform(0.6, 1.6))
        t = float(rs.uniform(1 / 365, 3.0))
        r = float(rs.uniform(-0.01, 0.08))
        sig = float(rs.uniform(0.05, 1.5))
        d1, d2 = quant_math._bs_d1_d2(s, k, t, r, sig)
        disc = math.exp(-r * t)
        ref_c = float(s * stats.norm.cdf(d1) - k * disc * stats.norm.cdf(d2))
        got_c = quant_math.bs_price(s, k, t, r, sig, "c")
        assert got_c is not None
        if abs(ref_c) > 0.01:
            assert abs(got_c - ref_c) / abs(ref_c) <= 1e-11
            n_kontrol += 1
    assert n_kontrol > 300          # kontrolün gerçekten koştuğunun kanıtı


def test_implied_vol_gidis_donus_korunur():
    """Brent döngüsü 1 ULP'lik farkı BÜYÜTEBİLİRDİ — ölçüldü: maks |Δσ| ~9e-8."""
    for sig in (0.08, 0.2, 0.45, 0.9):
        px = quant_math.bs_price(100.0, 105.0, 0.5, 0.03, sig, "c")
        assert px is not None
        iv = options_math.implied_vol(px, 100.0, 105.0, 0.5, 0.03, "c")
        assert iv is not None and abs(iv - sig) < 1e-6


def test_greeks_sozlesmesi_bozulmadi():
    """delta ∈ [0,1] (call), gamma ≥ 0, vega ≥ 0 — kütüphane değişimi işaretleri
    bozmamalı (karar bunlara bakar)."""
    g = quant_math.bs_greeks(100.0, 100.0, 0.5, 0.03, 0.25, "c")
    assert g is not None
    assert 0.0 <= g["delta"] <= 1.0 and g["gamma"] >= 0.0 and g["vega"] >= 0.0
    gp = quant_math.bs_greeks(100.0, 100.0, 0.5, 0.03, 0.25, "p")
    assert gp is not None and -1.0 <= gp["delta"] <= 0.0
    assert g["delta"] - gp["delta"] == pytest.approx(1.0, abs=1e-12)   # put-call parite
