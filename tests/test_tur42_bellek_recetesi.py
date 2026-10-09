"""Tur-42 — BELLEK REÇETESİ testleri: opt-in `optimizer_name="adafactor"` + `amp=True`.

İki bayrak da VARSAYILAN KAPALIDIR. Bu dosyanın BİRİNCİ işi bunu kilitlemektir: varsayılan
yol v8 ile BİT-DENK kalır ve yapılan seçim ARTEFAKTA yazılır (bir model hangi reçeteyle
eğitildiğini taşımadan karşılaştırmaya sokulamaz — Tur-22 model kartı kuralı).
İKİNCİ işi, bayraklar AÇIKKEN kırılan yeri korumaktır: bf16 autocast altında MoE
`index_add_` tip uyuşmazlığı. Bu hata CUDA GEREKTİRMEZ — CPU autocast'inde bf16 girdiyle
birebir üretilir, bu yüzden regresyon kilidi CPU-only ortamda da koşar.

Ağır testler küçük tutulur (`deep_mlp`, n=40, max_epochs<=2; CLI testinde --synthetic 150).
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import pytest

from nihai.ai import neural
from research import train_neural

TORCH = pytest.mark.skipif(not neural.TORCH_AVAILABLE, reason="torch yok")


@pytest.fixture()
def feat() -> list[str]:
    return [f"f{i}" for i in range(6)]


@pytest.fixture()
def veri(feat):
    rng = np.random.default_rng(42)
    n = 40
    X = rng.normal(size=(n, len(feat)))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def _fit(feat, veri, **kw):
    X, y = veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat, seed=7)
    m.fit(X, y, max_epochs=2, **kw)
    return m


# ---------------------------------------------------------------------------
# A. Seçim ARTEFAKTA yazılır (kart/manifest üzerinden izlenebilirlik)
# ---------------------------------------------------------------------------
@TORCH
def test_varsayilan_recete_adamw_ve_amp_kapali(feat, veri):
    rec = _fit(feat, veri).meta()["recipe"]
    assert rec["optimizer"] == "adamw"
    assert rec["amp"] is False


@TORCH
def test_adafactor_receteye_yazilir(feat, veri):
    rec = _fit(feat, veri, optimizer_name="adafactor").meta()["recipe"]
    assert rec["optimizer"] == "adafactor"


@TORCH
def test_optimizer_adi_bosluk_ve_buyuk_harfe_dayanikli(feat, veri):
    """Reçete bir DİZEDİR ve CLI/JSON'dan gelir; "  AdaFactor " sessizce AdamW'ye
    düşmemeli (sessiz düşüş = yanlış reçeteyle eğitildiği sanılan bir artefakt)."""
    rec = _fit(feat, veri, optimizer_name="  AdaFactor ").meta()["recipe"]
    assert rec["optimizer"] == "adafactor"


@TORCH
def test_optimizer_none_varsayilana_duser(feat, veri):
    assert _fit(feat, veri, optimizer_name=None).meta()["recipe"]["optimizer"] == "adamw"


@TORCH
def test_bilinmeyen_optimizer_valueerror(feat, veri):
    """Yazım hatası SESSİZCE AdamW'ye düşmemeli — düşseydi kullanıcı Adafactor ile
    eğittiğini sanır ve OOM'un nedenini anlamazdı (Tur-38'in "6 yanlış-yazım enforce'a
    düşmüyor" kuralının eğitim tarafı)."""
    with pytest.raises(ValueError, match="adamw|adafactor"):
        _fit(feat, veri, optimizer_name="adam")


# ---------------------------------------------------------------------------
# B. VARSAYILAN yol bit-denk (v8 parite iddiası)
# ---------------------------------------------------------------------------
@TORCH
def test_acik_adamw_varsayilan_ile_bit_denk(feat, veri):
    X, _ = veri
    p0 = _fit(feat, veri).predict_proba(X)
    p1 = _fit(feat, veri, optimizer_name="adamw", amp=False).predict_proba(X)
    assert np.array_equal(p0, p1)


@TORCH
def test_amp_cpuda_sessizce_kapanir_ve_bit_denk(feat, veri):
    """`amp=True` CPU'da bir HATA değildir ama bir ETKİ de değildir: bf16 autocast yalnız
    CUDA'da açılır. Kayıt bunu DÜRÜSTÇE söylemeli (`amp: False`) — "istendi" ile "oldu"
    aynı alana yazılırsa kart yalan söyler."""
    X, _ = veri
    m = _fit(feat, veri, amp=True)
    assert m.meta()["recipe"]["amp"] is False
    assert np.array_equal(m.predict_proba(X), _fit(feat, veri).predict_proba(X))


@TORCH
def test_amp_kapaliyken_autocast_baglami_hic_kurulmaz(feat, veri, monkeypatch):
    """Devre DIŞI bir `torch.autocast` bile toplu-iş başına 7,04 µs'dir (ölçüldü);
    VARSAYILAN yol bu vergiyi ödememeli. Ayrıca CUDA'sız kurulumda `autocast("cuda")`
    KURMAK uyarı bastırır. Bağlam artık döngü DIŞINDA ve yalnız amp açıkken kurulur."""
    cagri = {"n": 0}
    gercek = neural.torch.autocast

    def sayan(*a, **kw):
        cagri["n"] += 1
        return gercek(*a, **kw)

    monkeypatch.setattr(neural.torch, "autocast", sayan)
    _fit(feat, veri)
    assert cagri["n"] == 0


# ---------------------------------------------------------------------------
# C. Adafactor GERÇEKTEN eğitir — ve bedeli saklı değildir
# ---------------------------------------------------------------------------
@TORCH
def test_adafactor_egitir_ve_gecerli_olasilik_uretir(feat, veri):
    X, _ = veri
    p = _fit(feat, veri, optimizer_name="adafactor").predict_proba(X)
    assert p.shape == (len(X),)
    assert np.all(np.isfinite(p)) and np.all((p > 0.0) & (p < 1.0))


@TORCH
def test_adafactor_agirliklari_adamw_ile_ayni_degil(feat, veri):
    """BEDEL SAKLI DEĞİL: bu FARKLI bir optimizasyon algoritmasıdır, aynı reçetenin ucuz
    hâli değil. Bir gün bit-denk çıkarsa reçete GERÇEKTEN uygulanmıyor demektir."""
    X, _ = veri
    p_adamw = _fit(feat, veri).predict_proba(X)
    p_ada = _fit(feat, veri, optimizer_name="adafactor").predict_proba(X)
    assert not np.array_equal(p_adamw, p_ada)


@TORCH
def test_adafactor_optimizer_durumu_adamwden_kat_kat_kucuk():
    """Özelliğin VAROLUŞ GEREKÇESİ ölçülür: Adafactor ikinci momenti satır×sütun
    çarpanlarına ayırır ve birinci momenti hiç tutmaz → durum O(n) yerine O(√n).
    Bir torch yükseltmesi bunu sessizce değiştirirse (ör. momentum eklenirse) reçetenin
    tek gerekçesi çöker; bu test o sessiz değişimi yakalar."""
    import torch

    def durum_baytlari(yapici):
        torch.manual_seed(0)
        lin = torch.nn.Linear(512, 512, bias=False)
        opt = yapici(lin.parameters())
        lin(torch.zeros(2, 512)).sum().backward()
        opt.step()
        return sum(v.numel() * v.element_size()
                   for st in opt.state.values() for v in st.values()
                   if isinstance(v, torch.Tensor))

    adamw = durum_baytlari(lambda p: torch.optim.AdamW(p, lr=1e-3))
    ada = durum_baytlari(lambda p: torch.optim.Adafactor(p, lr=1e-3))
    assert adamw > 0
    assert ada * 10 < adamw, f"AdamW {adamw} B - Adafactor {ada} B: çarpanlaştırma yok?"


# ---------------------------------------------------------------------------
# D. bf16 autocast — MoE `index_add_` tip uyuşmazlığı (CUDA GEREKMEZ)
# ---------------------------------------------------------------------------
@TORCH
def test_index_add_tip_uyusmazligi_gercekten_hata_verir():
    """Aşağıdaki testin KORUDUĞU hatanın var olduğunu kanıtlar — cast'in kargo-kült
    olmadığını gösteren kontrol testi."""
    import torch

    out = torch.zeros(4, 3, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="same scalar type"):
        out.index_add_(0, torch.tensor([0, 2]), torch.ones(2, 3, dtype=torch.float32))


def _moe_kuruculari():
    if not neural.TORCH_AVAILABLE:
        return []
    return [
        lambda: neural._MoEFFN(d_model=8, ffn_dim=16, n_experts=4, top_k=2, dropout=0.0),
        lambda: neural._FineGrainedMoEFFN(d_model=8, expert_ffn=16, n_routed=4, top_k=2,
                                          shared_ffn=8, dropout=0.0),
    ]


@TORCH
@pytest.mark.parametrize("kur", _moe_kuruculari())
def test_moe_ffn_bf16_autocast_altinda_calisir(kur):
    """bf16 girdi altında `out` bf16, kapı olasılığı (softmax fp32 listesindedir) float32
    olur → çarpım float32'ye YÜKSELİR ve `index_add_` patlar. `.to(out.dtype)` bunu kapatır.
    Autocast CPU'da açılır: hata CUDA'ya değil TİP YÜKSELTMESİNE bağlıdır."""
    import torch

    torch.manual_seed(0)
    m = kur().eval()
    x = torch.randn(3, 5, 8, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()


@TORCH
@pytest.mark.parametrize("kur", _moe_kuruculari())
def test_moe_ffn_cast_amp_kapaliyken_no_op(kur):
    """`.to(out.dtype)` amp KAPALIYKEN NO-OP olmalı: float32 yol bit-denk kalır."""
    import torch

    torch.manual_seed(0)
    m = kur().eval()
    x = torch.randn(3, 5, 8)
    with torch.no_grad():
        y1 = m(x)
        y2 = m(x)
    assert y1.dtype == torch.float32
    assert torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# E. CLI KABLOSU — bayrak fit()'e ULAŞIYOR mu (yazılıp çağrılmayan bayrak kusurdur)
# ---------------------------------------------------------------------------
@TORCH
def test_cli_optimizer_bayragi_artefakta_ulasir(tmp_path):
    base = tmp_path / "cli_ada"
    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2", "--synthetic", "150",
                            "--max-epochs", "1", "--optimizer", "adafactor", "--amp",
                            "--save", str(base)])
    assert rc == 0
    rec = json.loads(Path(f"{base}.json").read_text(encoding="utf-8"))["recipe"]
    assert rec["optimizer"] == "adafactor"
    assert rec["amp"] is False       # CPU koşusu: istendi ama olmadı — kayıt dürüst


@TORCH
def test_cli_varsayilan_recete_satirinda_optimizer_gorunmez(tmp_path, capsys):
    """`adamw` bir SEÇİM değil VARSAYILANDIR: reçete özet satırı onu listelememeli,
    yoksa her koşu reçete değiştirmiş gibi görünür."""
    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2", "--synthetic", "120",
                            "--max-epochs", "1"])
    assert rc == 0
    ozet = [s for s in capsys.readouterr().out.splitlines() if "reçete=" in s]
    assert ozet, "reçete özet satırı basılmadı"
    assert "optimizer_name" not in ozet[0]
    assert "adamw" not in ozet[0]


def test_cli_gecersiz_optimizer_argparse_hatasi():
    with pytest.raises(SystemExit):
        train_neural.main(["--tier", "deep_mlp", "--synthetic", "100", "--optimizer", "adam"])


def test_contextlib_nullcontext_ardisik_yeniden_girilebilir():
    """fit() bağlamı döngü DIŞINDA bir kez kurar ve her toplu-işte yeniden girer;
    ardışık yeniden giriş güvenlidir (iç içe DEĞİL)."""
    ctx = contextlib.nullcontext()
    for _ in range(3):
        with ctx:
            pass
