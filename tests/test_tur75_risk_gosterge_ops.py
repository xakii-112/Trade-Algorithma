"""Tur-75 — `quant_math` risk eklentileri, üç yeni gösterge ve operasyon araçları.

Kapsanan maddeler: 107 (GARCH-M) · 108 (yarı-varyans) · 112 (Fisher) · 113 (Süper
Yumuşatıcı) · 114 (FRAMA) · 202 (MCR) · 203 (CoVaR) · 211 (EVaR) · 242 (saat kayması)
· 244 (hash zinciri) · 254 (Form 8949).

Ağ YOK: `clock_drift` testleri soketi monkeypatch'ler (guard zaten `conftest`te ama
ölçüm mantığı da kilitlenmeli — Tur-74 dersi: fail-soft bir yolu İSTİSNAYLA problamak
hiçbir şey kanıtlamaz, yan etkiyle problanır).
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pandas as pd
import pytest

from nihai import audit_chain as ac
from nihai import tax_report as tr
from nihai.ai import quant_math as qm
from nihai.signals import indicators_ext as ix


# ---------------------------------------------------------------------------
# 1) Madde 108 — gerçekleşen yarı-varyans
# ---------------------------------------------------------------------------
def test_yari_varyans_toplami_RV_ye_esittir():
    """RS⁻ + RS⁺ = RV — bu bir KİMLİKTİR, yaklaşım değil."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 5000)
    d = qm.realized_semivariance(r)
    assert d["rs_minus"] + d["rs_plus"] == pytest.approx(d["rv"], rel=1e-12)
    assert d["rv"] == pytest.approx(qm.realized_variance(r) if hasattr(qm, "realized_variance")
                                    else float(np.sum(r * r)), rel=1e-12)


def test_yari_varyans_asimetriyi_yakalar():
    """Yalnız aşağı hareket eden seride asimetri +1'e yakın olmalı."""
    r = np.r_[np.full(50, -0.02), np.full(50, 0.0)]
    d = qm.realized_semivariance(r)
    assert d["asymmetry"] == pytest.approx(1.0)
    assert d["n_zero"] == 50


def test_yari_varyans_duz_seride_asimetri_None():
    """RV = 0 → asimetri 0/0'dır; 0,0 basmak 'simetrik' demek olurdu."""
    d = qm.realized_semivariance(np.zeros(20))
    assert d["rv"] == 0.0 and d["asymmetry"] is None


# ---------------------------------------------------------------------------
# 2) Madde 107 — GARCH-in-Mean
# ---------------------------------------------------------------------------
def _garch_m_serisi(delta: float, n: int = 3000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s2 = np.zeros(n)
    r = np.zeros(n)
    s2[0] = 1e-4
    for t in range(1, n):
        s2[t] = 2e-6 + 0.10 * (r[t - 1] - delta * s2[t - 1]) ** 2 + 0.85 * s2[t - 1]
        r[t] = delta * s2[t] + np.sqrt(s2[t]) * rng.normal()
    return r


def test_garch_m_delta_SIFIRKEN_sifir_bulur_ve_LR_sifir():
    d = qm.garch_m_fit(_garch_m_serisi(0.0))
    assert abs(d["delta"]) < 2.0
    lr = 2.0 * (d["loglik"] - d["loglik_garch11"])
    assert -0.5 < lr < 4.0, "δ=0 iken LR χ²(1) tabanında olmalı"


def test_garch_m_pozitif_delta_isaretini_ve_anlamliligini_bulur():
    """Çoklu başlangıç OLMADAN bu test KIRMIZI olurdu (δ̂ sırta takılıp 0 kalıyordu)."""
    d = qm.garch_m_fit(_garch_m_serisi(30.0))
    assert d["delta"] > 5.0
    assert 2.0 * (d["loglik"] - d["loglik_garch11"]) > 3.84   # χ²(1) %95


def test_garch_m_negatif_delta_isaretini_bulur():
    assert qm.garch_m_fit(_garch_m_serisi(-30.0))["delta"] < -5.0


def test_garch_m_loglik_garch11_ile_AYNI_olcekte():
    """Sabit terim (ln 2π) atlanırsa LR n·ln(2π)/2 kadar kayar (ölçüldü: 3675,8)."""
    d = qm.garch_m_fit(_garch_m_serisi(0.0))
    assert abs(2.0 * (d["loglik"] - d["loglik_garch11"])) < 100.0


def test_garch_m_kisa_ve_sabit_seride_none():
    assert qm.garch_m_fit(np.zeros(50)) is None
    assert qm.garch_m_fit(np.full(500, 0.01)) is None


# ---------------------------------------------------------------------------
# 3) Madde 211 — EVaR
# ---------------------------------------------------------------------------
def test_evar_cvar_var_SIRALAMASI_yapisaldir():
    rng = np.random.default_rng(1)
    for dagilim in (rng.normal(0, 0.01, 3000), rng.standard_t(4, 3000) * 0.01):
        d = qm.evar(dagilim, 0.05)
        assert d["evar"] >= d["cvar"] >= d["var"], "EVaR ≥ CVaR ≥ VaR bozuldu"


def test_evar_alpha_kuyruk_olasiligi_konvansiyonunda():
    """α küçüldükçe (daha uç kuyruk) EVaR BÜYÜMELİ — BULGU-73-1'in tekrarına karşı."""
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 3000)
    assert qm.evar(r, 0.01)["evar"] > qm.evar(r, 0.10)["evar"]


def test_evar_duz_seride_sifir_ve_gecersizde_none():
    d = qm.evar(np.zeros(50))
    assert d["evar"] == 0.0 and d["z"] is None
    assert qm.evar(np.zeros(50), alpha=0.0) is None
    assert qm.evar([1.0], 0.05) is None


# ---------------------------------------------------------------------------
# 4) Madde 203 — CoVaR
# ---------------------------------------------------------------------------
def test_covar_bagimli_sistemde_dCoVaR_negatif():
    """Sistem varlığa 0,7 katsayıyla bağlıysa varlık sıkıntıdayken sistem de kötüdür."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.02, 2000)
    s = 0.7 * a + 0.3 * rng.normal(0, 0.02, 2000)
    d = qm.covar(a, s, 0.05)
    assert d["delta_covar"] < -0.01
    assert d["beta_tail"] == pytest.approx(0.7, abs=0.15)


def test_covar_bagimsiz_sistemde_dCoVaR_kucuk():
    """NEGATİF KONTROL: bağımsız serilerde ek kuyruk riski ~0."""
    rng = np.random.default_rng(5)
    d = qm.covar(rng.normal(0, 0.02, 2000), rng.normal(0, 0.02, 2000), 0.05)
    assert abs(d["delta_covar"]) < 0.005
    assert abs(d["beta_tail"]) < 0.15


def test_covar_gecersiz_alpha_ve_duz_varlikta_none():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 100)
    assert qm.covar(x, x, alpha=0.6) is None
    assert qm.covar(np.zeros(100), x) is None


# ---------------------------------------------------------------------------
# 5) Madde 202 — marjinal risk katkısı
# ---------------------------------------------------------------------------
def test_mcr_euler_kimligi_makine_hassasiyetinde():
    """Σ CR_i = σ_p — Euler'in homojen fonksiyon teoremi, YAPISAL."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (8, 8))
    S = A @ A.T / 8.0
    w = rng.random(8)
    w /= w.sum()
    d = qm.marginal_risk_contributions(w, S)
    assert float(d["component_risk"].sum()) == pytest.approx(d["portfolio_vol"], rel=1e-12)
    assert float(d["share"].sum()) == pytest.approx(1.0, rel=1e-12)


def test_mcr_agirlik_payi_ile_risk_payi_AYRISIR():
    """Madde 202'nin var oluş nedeni: küçük ağırlık BÜYÜK risk payı üretebilir."""
    S = np.array([[0.0004, 0.0], [0.0, 0.25]])       # 2. varlık 25 kat oynak
    w = np.array([0.9, 0.1])
    d = qm.marginal_risk_contributions(w, S)
    assert w[1] == 0.1
    # %10 ağırlık, riskin %88'i: iki pay AYNI sayı değildir ve olmamalıdır.
    assert d["share"][1] > 0.85 > w[1], "risk payı ağırlık payıyla karıştırılmış"
    assert d["share"][1] / w[1] > 8.0
    assert d["max_share_idx"] == 1


def test_mcr_gecersiz_girdide_none():
    assert qm.marginal_risk_contributions([1.0, 0.0], np.zeros((2, 2))) is None
    assert qm.marginal_risk_contributions([1.0], np.eye(2)) is None


# ---------------------------------------------------------------------------
# 6) Madde 112/113/114 — üç gösterge
# ---------------------------------------------------------------------------
def _df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "volume": rng.uniform(1e5, 5e5, n)})


def test_fisher_duz_seride_TAM_SIFIR():
    """Sıfır bilgi → sıfır sinyal (BULGU-33-1). Kırpmasız uygulama ±inf üretirdi."""
    d = pd.DataFrame({"high": [10.0] * 60, "low": [10.0] * 60, "close": [10.0] * 60})
    f, t = ix.fisher_transform(d)
    assert float(f.iloc[-1]) == 0.0
    assert np.isfinite(f).all()


def test_fisher_sonlu_kalir_ve_tetik_bir_bar_gecikmelidir():
    f, t = ix.fisher_transform(_df())
    assert np.isfinite(f).all()
    assert t.iloc[-1] == pytest.approx(f.iloc[-2])


def test_super_smoother_gurultuyu_sonlu_kazancla_suzer():
    """Filtre çıktısının oynaklığı girdinin altında, ama seviyeyi takip etmeli."""
    rng = np.random.default_rng(3)
    temiz = np.linspace(100, 120, 400)
    gur = pd.Series(temiz + rng.normal(0, 2.0, 400))
    y = ix.super_smoother(gur, 10)
    assert float(np.std(np.diff(y[20:]))) < float(np.std(np.diff(gur[20:])))
    assert float(np.abs(y[50:] - temiz[50:]).mean()) < 1.0


def test_super_smoother_sabit_seride_sabittir():
    """DC kazancı 1 olmalı: c1 + c2 + c3 = 1 (aksi hâlde seviye kayardı)."""
    y = ix.super_smoother(pd.Series([42.0] * 100), 10)
    assert float(np.abs(y - 42.0).max()) < 1e-9


def test_frama_trendde_hizli_yatayda_yavas():
    """Fraktal boyut D→1 (trend) hızlı, D→2 (gürültü) yavaş ortalama üretmeli."""
    rng = np.random.default_rng(4)
    n = 400
    trend = np.linspace(100, 200, n)
    yatay = 100 + rng.normal(0, 1.0, n)
    d_tr = pd.DataFrame({"high": trend + 0.2, "low": trend - 0.2, "close": trend})
    d_ya = pd.DataFrame({"high": yatay + 0.2, "low": yatay - 0.2, "close": yatay})
    gecikme_tr = float(np.abs(ix.frama(d_tr)[50:] - trend[50:]).mean())
    gecikme_ya = float(np.abs(ix.frama(d_ya)[50:] - yatay[50:]).mean())
    assert gecikme_tr < gecikme_ya, "FRAMA trendde yavaş, gürültüde hızlı kalmış"


def test_frama_tek_periyot_cifte_yuvarlanir():
    d = _df(120)
    assert ix.frama(d, 15).name == "frama16"


def test_yeni_gostergeler_PANELE_bagli():
    """Yazılıp panele bağlanmayan gösterge bu depoda kusurdur (Tur-38 kuralı)."""
    df = _df()
    adlar = {r["name"] for r in ix.extended_rows(df, float(df["close"].iloc[-1]))}
    assert {"Fisher Transform", "FRAMA(16)", "SuperSmoother(10)"} <= adlar


# ---------------------------------------------------------------------------
# 7) Madde 244 — hash zinciri
# ---------------------------------------------------------------------------
def test_zincir_saglam_ve_merkle_kok_deterministik(tmp_path):
    p = tmp_path / "z.jsonl"
    for i in range(6):
        ac.append({"sembol": "AAPL", "i": i}, p)
    d = ac.verify(p)
    assert d["ok"] is True and d["n"] == 6
    assert ac.merkle_root(p) == ac.merkle_root(p)
    assert len(ac.merkle_root(p)) == 64


def test_zincir_ORTADAKI_kurcalamayi_TAM_SATIRINDA_yakalar(tmp_path):
    p = tmp_path / "z.jsonl"
    for i in range(5):
        ac.append({"karar": "WAIT", "i": i}, p)
    satirlar = p.read_text(encoding="utf-8").splitlines()
    k = json.loads(satirlar[2])
    k["payload"]["karar"] = "LONG"
    satirlar[2] = json.dumps(k, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    p.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    d = ac.verify(p)
    assert d["ok"] is False and d["broken_at"] == 2 and d["reason"] == "HASH_UYUSMUYOR"


def test_zincir_zaman_damgasi_da_muhurlu(tmp_path):
    """`ts` özete dahildir — damgayı değiştirmek de zinciri kırmalı."""
    p = tmp_path / "z.jsonl"
    for i in range(3):
        ac.append({"i": i}, p)
    satirlar = p.read_text(encoding="utf-8").splitlines()
    k = json.loads(satirlar[1])
    k["ts"] = "2000-01-01T00:00:00.000+00:00"
    satirlar[1] = json.dumps(k, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    p.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    assert ac.verify(p)["ok"] is False


def test_zincir_ayirac_cakismayi_engeller():
    """Ayıraçsız SHA-256(a||b) kurgusunda ("ab","c") ile ("a","bc") ÇAKIŞIR."""
    assert ac._digest("ab", "c") != ac._digest("a", "bc")


def test_zincir_dosya_yoksa_saglam_sayilir(tmp_path):
    d = ac.verify(tmp_path / "yok.jsonl")
    assert d["ok"] is True and d["n"] == 0 and d["reason"] == "DOSYA_YOK"
    assert ac.merkle_root(tmp_path / "yok.jsonl") is None


def test_zincir_seal_dogrulama_ve_kok_birlikte(tmp_path):
    p = tmp_path / "z.jsonl"
    ac.append({"i": 0}, p)
    d = ac.seal(p)
    assert d["ok"] is True and d["merkle_root"] is not None


# ---------------------------------------------------------------------------
# 8) Madde 242 — NTP saat kayması (sahte soketle, AĞSIZ)
# ---------------------------------------------------------------------------
def _sahte_ntp(monkeypatch, t2: float, t3: float, gonderildi: list | None = None):
    """T2/T3'ü paketin DOĞRU ofsetlerine (32-47) yazan sahte NTP sunucusu."""
    class _S:
        def __init__(self, *a, **k):
            pass

        def settimeout(self, v):
            pass

        def sendto(self, paket, adres):
            if gonderildi is not None:
                gonderildi.append(adres)

        def recvfrom(self, n):
            buf = bytearray(48)
            for ofs, t in ((32, t2), (40, t3)):
                sec = int(t) + ac.NTP_EPOCH_DELTA
                frac = int((t - int(t)) * 2 ** 32)
                buf[ofs:ofs + 8] = struct.pack("!II", sec, frac)
            return bytes(buf), ("1.2.3.4", 123)

        def close(self):
            pass

    monkeypatch.setattr(ac.socket, "socket", _S)


def test_saat_kaymasi_gecikme_duzeltmeli_hesaplanir(monkeypatch):
    """Sunucu tam doğru saatteyse ama tur 100 ms sürüyorsa offset 0 OLMALI.

    Tek yönlü farkı offset sayan (yaygın) uygulama 50 ms'lik hayalet kayma raporlar.
    """
    t = [1_000_000.000, 1_000_000.100]           # T1, T4 → 100 ms tur
    monkeypatch.setattr(ac.time, "time", lambda: t.pop(0))
    _sahte_ntp(monkeypatch, t2=1_000_000.050, t3=1_000_000.050)
    d = ac.clock_drift()
    assert d["measured"] is True
    assert d["offset_ms"] == pytest.approx(0.0, abs=1e-6)
    assert d["roundtrip_ms"] == pytest.approx(100.0, abs=1e-6)
    assert d["warn"] is False


def test_saat_kaymasi_gercek_kaymayi_bulur_ve_uyarir(monkeypatch):
    t = [1_000_000.000, 1_000_000.100]
    monkeypatch.setattr(ac.time, "time", lambda: t.pop(0))
    # Sunucu 500 ms İLERİDE: T2 = T1 + 0,05 + 0,5
    _sahte_ntp(monkeypatch, t2=1_000_000.550, t3=1_000_000.550)
    d = ac.clock_drift()
    assert d["offset_ms"] == pytest.approx(500.0, abs=1e-6)
    assert d["warn"] is True


def test_saat_kaymasi_ag_yokken_SIFIR_DONMEZ(monkeypatch):
    class _S:
        def __init__(self, *a, **k):
            pass

        def settimeout(self, v):
            pass

        def sendto(self, *a):
            raise OSError("ag yok")

        def recvfrom(self, n):
            raise OSError("ag yok")

        def close(self):
            pass

    monkeypatch.setattr(ac.socket, "socket", _S)
    d = ac.clock_drift()
    assert d["measured"] is False and d["offset_ms"] is None and d["warn"] is False


def test_metrics_olculmemis_saati_BASMAZ(monkeypatch):
    """`/metrics`te 0 ms 'kayma yok' demektir; ölçülmediyse satır HİÇ basılmamalı."""
    from nihai import observability as ob
    monkeypatch.setattr(ob, "_CLOCK", None)
    assert not [x for x in ob.prometheus_text().splitlines() if x.startswith("nihai_clock")]
    monkeypatch.setattr(ob, "_CLOCK", {"measured": True, "offset_ms": -12.5,
                                       "roundtrip_ms": 40.0, "warn": False})
    satirlar = [x for x in ob.prometheus_text().splitlines() if x.startswith("nihai_clock")]
    assert "nihai_clock_offset_ms -12.5" in satirlar
    assert "nihai_clock_drift_warn 0" in satirlar


def test_metrics_okuma_yolu_AG_CAGRISI_ODEMEZ(monkeypatch):
    """`prometheus_text` asla `clock_drift`i ÇAĞIRMAMALI (Tur-59 BULGU-59-8)."""
    from nihai import observability as ob
    cagrildi: list[int] = []
    monkeypatch.setattr(ac, "clock_drift", lambda *a, **k: cagrildi.append(1))
    ob.prometheus_text()
    ob.snapshot()
    assert cagrildi == []


# ---------------------------------------------------------------------------
# 9) Madde 254 — Form 8949
# ---------------------------------------------------------------------------
def _trades(tmp_path, rows: list[dict]):
    p = tmp_path / "trades.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_form8949_long_hasilat_ve_maliyeti_dogru_yerlestirir(tmp_path):
    p = _trades(tmp_path, [{
        "symbol": "AAPL", "asset_type": "stock", "side": "LONG",
        "entry_time": "2025-01-10", "exit_time": "2025-03-10",
        "entry_price": 100.0, "exit_price": 110.0, "quantity": 50,
        "realized_pnl": 500.0}])
    r = tr.build(p)
    s = r["rows"][0]
    assert s["proceeds"] == 5500.0 and s["cost_basis"] == 5000.0
    assert s["gain_loss"] == 500.0 and s["part"].startswith("I ")


def test_form8949_SHORT_hasilat_girisTEN_maliyet_cikisTAN(tmp_path):
    """Açığa satışta satış ÖNCE olur; ters yerleştiren uygulama işareti çevirir."""
    p = _trades(tmp_path, [{
        "symbol": "TSLA", "asset_type": "stock", "side": "SHORT",
        "entry_time": "2025-01-10", "exit_time": "2025-02-10",
        "entry_price": 200.0, "exit_price": 180.0, "quantity": 10,
        "realized_pnl": 200.0}])
    s = tr.build(p)["rows"][0]
    assert s["proceeds"] == 2000.0 and s["cost_basis"] == 1800.0
    assert s["gain_loss"] == 200.0


def test_form8949_1233_SHORT_her_zaman_kisa_vadelidir(tmp_path):
    """3 yıl tutulmuş bir SHORT bile Part I'dir (26 U.S.C. §1233(b))."""
    p = _trades(tmp_path, [{
        "symbol": "TSLA", "asset_type": "stock", "side": "SHORT",
        "entry_time": "2021-01-10", "exit_time": "2024-06-10",
        "entry_price": 200.0, "exit_price": 180.0, "quantity": 10,
        "realized_pnl": 200.0}])
    s = tr.build(p)["rows"][0]
    assert s["holding_days"] > 365 and s["part"].startswith("I ")


def test_form8949_uzun_vade_siniri_365_gunden_FAZLADIR(tmp_path):
    ortak = {"symbol": "MSFT", "asset_type": "stock", "side": "LONG",
             "entry_price": 100.0, "exit_price": 110.0, "quantity": 1,
             "realized_pnl": 10.0}
    p = _trades(tmp_path, [
        {**ortak, "entry_time": "2024-01-01", "exit_time": "2025-01-01"},   # 366 gün
        {**ortak, "entry_time": "2024-01-02", "exit_time": "2025-01-01"},   # 365 gün
    ])
    parts = [x["part"] for x in tr.build(p)["rows"]]
    assert parts[0].startswith("II") and parts[1].startswith("I ")


def test_form8949_1256_satirlari_AYRI_kovada(tmp_path):
    """Nakit uzlaşılı endeks opsiyonu Form 6781'e girer; 8949'a YAZILMAZ."""
    ortak = {"asset_type": "option", "side": "LONG", "entry_time": "2025-01-10",
             "exit_time": "2025-02-10", "entry_price": 5.0, "exit_price": 7.0,
             "quantity": 2, "realized_pnl": 400.0}
    p = _trades(tmp_path, [{**ortak, "symbol": "SPX"}, {**ortak, "symbol": "AAPL"}])
    r = tr.build(p)
    assert r["n"] == 1 and r["rows"][0]["symbol"] == "AAPL"
    assert r["n_1256"] == 1 and "6781" in r["section_1256"][0]["part"]


def test_form8949_wash_adayi_isaretlenir_ama_DUZELTME_yazilmaz(tmp_path):
    ortak = {"symbol": "NVDA", "asset_type": "stock", "side": "LONG", "quantity": 10}
    p = _trades(tmp_path, [
        {**ortak, "entry_time": "2025-01-05", "exit_time": "2025-01-20",
         "entry_price": 100.0, "exit_price": 90.0, "realized_pnl": -100.0},
        {**ortak, "entry_time": "2025-01-25", "exit_time": "2025-02-20",
         "entry_price": 92.0, "exit_price": 99.0, "realized_pnl": 70.0},
    ])
    r = tr.build(p)
    assert r["n_wash_candidates"] == 1
    assert all(x["adjustment_code"] == "" and x["adjustment_amount"] == 0.0
               for x in r["rows"]), "hesaplanmamış bir düzeltme yazılmış"


def test_form8949_dosya_yoksa_ve_kolon_eksikse_fail_soft(tmp_path):
    assert tr.build(tmp_path / "yok.csv")["reason"] == "DOSYA_YOK"
    p = tmp_path / "eksik.csv"
    pd.DataFrame({"symbol": ["AAPL"]}).to_csv(p, index=False)
    assert tr.build(p)["reason"].startswith("KOLON_EKSIK")


def test_form8949_csv_yazimi_ve_ayri_6781_dosyasi(tmp_path):
    ortak = {"asset_type": "option", "side": "LONG", "entry_time": "2025-01-10",
             "exit_time": "2025-02-10", "entry_price": 5.0, "exit_price": 7.0,
             "quantity": 2, "realized_pnl": 400.0}
    p = _trades(tmp_path, [{**ortak, "symbol": "SPX"}, {**ortak, "symbol": "AAPL"}])
    out = tmp_path / "f8949.csv"
    tr.write_csv(tr.build(p), out)
    assert out.exists()
    alt = out.with_name("f8949_6781.csv")
    assert alt.exists()
    assert list(pd.read_csv(out).columns) == tr.FORM_8949_COLS
    assert len(pd.read_csv(alt)) == 1
