"""Tur-36 (AI v8) yetenek testleri — MEGA basamağı · opt-in ek başlıklar (`extra_heads`) ·
kanıt-temelli yön (Dirichlet/EDL) · K-uzmanlı kaynak füzyonu · etiketsiz ön-eğitim seti.

Ağır testler küçük tutulur (mini n, max_epochs<=2); 2,01 MİLYARLIK gerçek yapılandırma
YALNIZ `theoretical_param_count` ile ölçülür (`meta` aygıtı → bellek ayırmaz).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nihai.ai import learning, models, neural

TORCH = pytest.mark.skipif(not neural.TORCH_AVAILABLE, reason="torch yok")


@pytest.fixture()
def feat() -> list[str]:
    return [f"f{i}" for i in range(6)]


def _mini(cls, names, **kw):
    """Küçük transformer örneği: mimari sözleşmesi korunur, gerçek ölçek testte koşmaz.
    Basamakların FFN argümanları farklıdır (ffn_dim ↔ expert_ffn/shared_ffn) — tek
    yardımcı hepsini kapsasın diye burada eşlenir."""
    small = dict(d_model=32, n_blocks=2, n_heads=2)
    if cls in (neural.QuantTransformerMoEXL, neural.QuantTransformerUltra,
               neural.QuantTransformerMega):
        small |= dict(expert_ffn=16, n_routed=4, top_k=2, shared_ffn=16)
    elif cls is neural.QuantTransformerMoE:
        small |= dict(ffn_dim=16, n_experts=4, top_k=2)
    else:
        small |= dict(ffn_dim=16)
    return cls(names, **small, **kw)


# ---------------------------------------------------------------------------
# 1. Kapasite merdiveni — v8 basamağı
# ---------------------------------------------------------------------------
@TORCH
def test_mega_param_sozlesmesi():
    """v8 SÖZLEŞMESİ: `quant_transformer_mega` >= 1,9 milyar ve ULTRA'nın ≈2,6 katı.
    Taban `n_features`'tan BAĞIMSIZDIR (blok yığını tek başına ≈2,0 milyar)."""
    p_mega = neural.theoretical_param_count("quant_transformer_mega", 8)
    p_ultra = neural.theoretical_param_count("quant_transformer_ultra", 8)
    assert p_mega is not None and p_ultra is not None
    assert p_mega >= 1_900_000_000
    assert 2.3 <= p_mega / p_ultra <= 3.0


def test_recommend_tier_mega_esigi():
    """26M satır eşiği: ÖNERİ, izin değil (promotion yine kanıt-kapılı ayrı yol)."""
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")
    assert models.recommend_tier(26_000_000, 96, allow_neural=True) == "quant_transformer_mega"
    assert models.recommend_tier(25_999_999, 96, allow_neural=True) == "quant_transformer_ultra"
    # allow_neural=False → v2 davranışı BİT-DENK.
    assert models.recommend_tier(26_000_000, 96) == "mlp"


@TORCH
def test_param_sayimi_kisa_yol_tam_kurulumla_ayni():
    """`_numel_of_built_model` ince-taneli MoE basamaklarında uzmanların TAMAMINI
    materyalize etmez (MEGA'da 1280 uzman = 1,62 s); `n_routed = top_k` ile kurar ve
    eksik uzmanları ÖLÇÜLEN uzman boyutuyla ekler.

    Bu test kısa yolun TAM kurulumla BİREBİR aynı sayıyı verdiğini kilitler — aksi hâlde
    optimizasyon sessizce yanlış bir parametre sayısı raporlardı (ve "parametre arttı"
    iddiası ölçüm değil temenni olurdu)."""
    import torch
    for tier in ("quant_transformer_moe_xl", "quant_transformer_ultra"):
        with torch.device("meta"):
            tam = int(sum(p.numel()
                          for p in neural._build_torch_model(tier, ["f0", "f1"], 42).parameters()))
        assert neural.theoretical_param_count(tier, 2) == tam
    # MEGA'nın tam kurulumu testte 1,6 s yerdi → sayı DONDURULDU (mimari bilinçli
    # değişirse bu satır da bilinçli güncellenir; sessiz sapma imkânsız).
    assert neural.theoretical_param_count("quant_transformer_mega", 8) == 2_010_698_010
    assert neural.theoretical_param_count("quant_transformer_ultra", 8) == 768_999_640


def test_neural_tiers_literal_kopyasi_hala_esit():
    """models.py NEURAL_TIERS'ı LİTERAL kopyalar (torch importunu tetiklememek için) —
    yeni basamak eklenince iki liste ayrışmamalı."""
    assert models.NEURAL_TIERS == neural.NEURAL_TIERS


# ---------------------------------------------------------------------------
# 2. Opt-in ek başlıklar — VARSAYILAN geriye uyum
# ---------------------------------------------------------------------------
@TORCH
def test_varsayilan_extra_heads_v7_ile_ayni(feat):
    """`extra_heads` VERİLMEZSE: alt basamaklarda ek başlık YOK, ULTRA'da yalnız `vol_q`,
    MEGA'da ikisi de. Bu, kayıtlı v7 artefaktlarının bit-denk yüklenmesinin dayanağıdır."""
    import torch
    x = torch.zeros(2, len(feat))
    for cls in (neural.QuantTransformer, neural.QuantTransformerXL,
                neural.QuantTransformerMoE, neural.QuantTransformerMoEXL):
        out = _mini(cls, feat).eval()(x)
        assert "vol_q" not in out and "dir_evid" not in out
    ultra = _mini(neural.QuantTransformerUltra, feat).eval()(x)
    assert "vol_q" in ultra and "dir_evid" not in ultra
    mega = _mini(neural.QuantTransformerMega, feat).eval()(x)
    assert "vol_q" in mega and "dir_evid" in mega
    assert "dir_evid" not in neural.DeepMLP(feat).eval()(x)


@TORCH
def test_extra_head_sirasi_onemsiz_agirliklar_ayni(feat):
    """Ek başlık kurulum sırası KANONİKTİR: çağıran sırayı değiştirince RNG akışı ve
    dolayısıyla AĞIRLIKLAR değişmemeli (aynı küme = aynı model)."""
    import torch
    torch.manual_seed(5)
    a = _mini(neural.QuantTransformerXL, feat, extra_heads=("vol_q", "dir_evid"))
    torch.manual_seed(5)
    b = _mini(neural.QuantTransformerXL, feat, extra_heads=("dir_evid", "vol_q"))
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


@TORCH
def test_bilinmeyen_extra_head_hata(feat):
    """Sessiz atlama YOK: yazım hatası hatadır (aksi hâlde 'başlık eklendi sandım' olur)."""
    with pytest.raises(ValueError, match="extra_head"):
        _mini(neural.QuantTransformerXL, feat, extra_heads=("volq",))
    with pytest.raises(ValueError, match="extra_head"):
        neural.NeuralCapacity("deep_mlp", feat, extra_heads=("dir_evidence",))


@TORCH
def test_volq_alt_basamakta_opt_in_calisir(feat):
    """Tur-35 açık kalem #2: `vol_q` artık ULTRA'ya gömülü değil — istenen basamakta
    açılabilir ve AZALMAYAN çıkar (kümülatif softplus kurulumu)."""
    import torch
    m = _mini(neural.QuantTransformerXL, feat, extra_heads=("vol_q",)).eval()
    q = m(torch.randn(8, len(feat)))["vol_q"]
    assert q.shape == (8, len(neural.VOL_QUANTILES))
    assert bool((q[:, 1:] >= q[:, :-1]).all())


# ---------------------------------------------------------------------------
# 3. Kanıt-temelli yön (Dirichlet / EDL)
# ---------------------------------------------------------------------------
@TORCH
def test_dirichlet_alpha_ve_belirsizlik_sozlesmesi(feat):
    """α ≥ 1 (kanıt e = α−1 ≥ 0) ve u = K/S ∈ (0, 1]. u=1 tam olarak 'hiç kanıt yok'."""
    import torch
    m = _mini(neural.QuantTransformerMega, feat).eval()
    alpha = m(torch.randn(16, len(feat)))["dir_evid"]
    assert alpha.shape == (16, 2)
    assert bool((alpha >= 1.0).all())
    u = 2.0 / alpha.sum(dim=1)
    assert bool(((u > 0) & (u <= 1.0)).all())
    # Sıfır ham çıktı → softplus(0)=ln2 → α=1+ln2 → u<1; tam 1 yalnız ham→−∞ limitinde.
    assert float(neural._dirichlet_alpha(torch.zeros(1, 2))[0, 0]) == pytest.approx(
        1.0 + math.log(2.0), abs=1e-6)


@TORCH
def test_evidential_loss_yanlis_kesinligi_daha_cok_cezalandirir():
    """Kaybın YÖNÜ: doğru sınıfta bol kanıt < kanıtsız < YANLIŞ sınıfta bol kanıt."""
    import torch
    y = torch.tensor([1])
    dogru = torch.tensor([[1.0, 21.0]])      # sınıf 1'e 20 kanıt
    kanitsiz = torch.tensor([[1.0, 1.0]])    # düz Dirichlet
    yanlis = torch.tensor([[21.0, 1.0]])
    la = float(neural._evidential_loss(dogru, y))
    lb = float(neural._evidential_loss(kanitsiz, y))
    lc = float(neural._evidential_loss(yanlis, y))
    assert la < lb < lc


@TORCH
def test_evidential_loss_anneal_sifirda_kl_yok():
    """anneal=0 → YALNIZ Bayes-riski CE. Tavlama katsayısı gerçekten devrede olmalı;
    aksi hâlde ilk epoklarda kanıt oluşmadan cezalanır (Sensoy §4)."""
    import torch
    y = torch.tensor([1, 0])
    alpha = torch.tensor([[2.0, 9.0], [7.0, 3.0]])
    l0 = float(neural._evidential_loss(alpha, y, anneal=0.0))
    l1 = float(neural._evidential_loss(alpha, y, anneal=1.0))
    assert l1 > l0            # KL ≥ 0 ve bu örnekte yanıltıcı kanıt var


@TORCH
def test_predict_evidential_baslik_yoksa_none(feat):
    """Başlığı olmayan modelde 'belirsizlik yok' DEĞİL, 'bu mimaride ölçülmüyor' → None."""
    rs = np.random.RandomState(0)
    X, y = rs.rand(40, len(feat)), (rs.rand(40) > 0.5).astype(int)
    m = neural.NeuralCapacity("deep_mlp", feat, seed=1)
    with pytest.raises(RuntimeError):
        m.predict_evidential(X)                     # fit edilmemiş
    m.fit(X, y, max_epochs=2, patience=1)
    assert m.predict_evidential(X) is None
    assert m.effective_extra_heads() == []


@TORCH
def test_predict_evidential_p_ve_u_donuyor(feat):
    rs = np.random.RandomState(1)
    X, y = rs.rand(60, len(feat)), (rs.rand(60) > 0.5).astype(int)
    m = neural.NeuralCapacity("deep_mlp", feat, seed=1, extra_heads=("dir_evid",))
    m.fit(X, y, max_epochs=2, patience=1)
    res = m.predict_evidential(X)
    assert res is not None
    p, u = res
    assert p.shape == u.shape == (60,)
    assert bool(((p >= 0) & (p <= 1)).all())
    assert bool(((u > 0) & (u <= 1)).all())
    assert m.effective_extra_heads() == ["dir_evid"]


@TORCH
def test_extra_head_ana_gorevi_ele_gecirmez(feat):
    """`predict_proba` HÂLÂ `direction` başlığını okur (kanıt başlığı İKİNCİ bir ölçüdür,
    ikinci bir GÖRÜŞ değil) — ve sıcaklık kalibrasyonu ona uygulanmaz."""
    rs = np.random.RandomState(2)
    X, y = rs.rand(60, len(feat)), (rs.rand(60) > 0.5).astype(int)
    m = neural.NeuralCapacity("deep_mlp", feat, seed=3, extra_heads=("dir_evid",))
    m.fit(X, y, max_epochs=2, patience=1)
    p_dir = m.predict_proba(X)
    p_ev, _ = m.predict_evidential(X)               # type: ignore[misc]
    assert p_dir.shape == p_ev.shape
    assert not np.allclose(p_dir, p_ev)             # AYNI sayı değiller (ayrı başlıklar)


@TORCH
def test_topluluk_kanit_konsantrasyonda_birlestirir(tmp_path, feat):
    """Topluluk α'ları ORTALAR, p/u oradan türetilir — p ve u'yu ayrı ayrı ortalamak
    tutarsız olurdu (u = K/Σα doğrusal DEĞİLDİR). Başlıksız üyede dürüstçe None.
    `save`/`load_any` yeteneği geri getirmeli."""
    rs = np.random.RandomState(6)
    X, y = rs.rand(60, len(feat)), (rs.rand(60) > 0.5).astype(int)
    ens = neural.NeuralEnsemble("deep_mlp", feat, seed=3, n_members=2,
                                extra_heads=("dir_evid",))
    ens.fit(X, y, max_epochs=2, patience=1)
    res = ens.predict_evidential(X)
    assert res is not None
    p, u = res
    assert p.shape == u.shape == (60,)
    assert bool(((u > 0) & (u <= 1)).all())
    base = neural.NeuralEnsemble("deep_mlp", feat, seed=3, n_members=2)
    base.fit(X, y, max_epochs=2, patience=1)
    assert base.predict_evidential(X) is None
    ens.save(tmp_path / "ens")
    back = neural.load_any(tmp_path / "ens")
    assert isinstance(back, neural.NeuralEnsemble)
    back_res = back.predict_evidential(X)
    assert back_res is not None and np.allclose(back_res[1], u)


# ---------------------------------------------------------------------------
# 4. Kalıcılık — extra_heads artefaktla taşınır, ESKİ artefakt bozulmaz
# ---------------------------------------------------------------------------
@TORCH
def test_save_load_extra_heads_tasir(tmp_path, feat):
    rs = np.random.RandomState(3)
    X, y = rs.rand(50, len(feat)), (rs.rand(50) > 0.5).astype(int)
    m = neural.NeuralCapacity("deep_mlp", feat, seed=4, extra_heads=("vol_q", "dir_evid"))
    m.fit(X, y, aux={"vol": rs.rand(50)}, max_epochs=2, patience=1)
    base = tmp_path / "m"
    m.save(base)
    back = neural.NeuralCapacity.load(base)
    assert back is not None
    assert back.effective_extra_heads() == ["vol_q", "dir_evid"]
    assert back.meta()["extra_heads"] == ["vol_q", "dir_evid"]
    assert np.allclose(back.predict_proba(X), m.predict_proba(X))


@TORCH
def test_extra_heads_alani_olmayan_eski_artefakt_yuklenir(tmp_path, feat):
    """v7 ve öncesi JSON'larda `extra_heads` YOKTUR → basamak varsayılanına düşmeli."""
    rs = np.random.RandomState(4)
    X, y = rs.rand(40, len(feat)), (rs.rand(40) > 0.5).astype(int)
    m = neural.NeuralCapacity("deep_mlp", feat, seed=5)
    m.fit(X, y, max_epochs=2, patience=1)
    base = tmp_path / "eski"
    m.save(base)
    raw = json.loads((tmp_path / "eski.json").read_text(encoding="utf-8"))
    raw.pop("extra_heads")
    (tmp_path / "eski.json").write_text(json.dumps(raw), encoding="utf-8")
    back = neural.NeuralCapacity.load(base)
    assert back is not None and back.is_fitted
    assert np.allclose(back.predict_proba(X), m.predict_proba(X))


# ---------------------------------------------------------------------------
# 5. K-uzmanlı kaynak füzyonu (Tur-35 açık kalem #4)
# ---------------------------------------------------------------------------
def test_source_fusion_ek_uzman_isimleri_hizali():
    rs = np.random.RandomState(5)
    n = 300
    y = (rs.rand(n) > 0.5).astype(int)
    res = learning.source_fusion(
        list(np.clip(rs.rand(n), 0.05, 0.95)), list(np.clip(rs.rand(n), 0.05, 0.95)),
        list(y), extra={"prob_cal": list(np.clip(rs.rand(n), 0.05, 0.95))})
    assert res is not None
    assert res["experts"] == ["pwin", "neural", "prob_cal", "base"]
    assert len(res["loss_experts"]) == len(res["final_weights"]) == 4
    assert res["best_expert"] in res["experts"]


def test_source_fusion_hizasiz_ek_uzman_none():
    """Sessiz kırpma YOK: farklı uzunluktaki uzman = anlamsız birleşim → None."""
    rs = np.random.RandomState(6)
    n = 50
    assert learning.source_fusion(list(rs.rand(n)), list(rs.rand(n)),
                                  list((rs.rand(n) > 0.5).astype(int)),
                                  extra={"x": list(rs.rand(n - 1))}) is None


def test_source_fusion_ek_uzmansiz_cagri_tur35_ile_ayni():
    """`extra` verilmezse çıktı Tur-35 ile AYNI (üç uzman, aynı sıra)."""
    rs = np.random.RandomState(7)
    n = 300
    y = (rs.rand(n) > 0.5).astype(int)
    res = learning.source_fusion(list(np.clip(rs.rand(n), 0.05, 0.95)),
                                 list(np.clip(rs.rand(n), 0.05, 0.95)), list(y))
    assert res is not None and res["experts"] == ["pwin", "neural", "base"]


# ---------------------------------------------------------------------------
# 6. Etiketsiz ön-eğitim seti (Tur-35 açık kalem #3)
# ---------------------------------------------------------------------------
def test_build_unlabeled_cozulmus_satiri_disliyor(tmp_path):
    """Etiketsiz = özellik logunda VAR, çözülmüş tahminde YOK."""
    import pandas as pd

    from nihai.signals.meta_features import FEATURE_NAMES
    from research import build_live_dataset as bld

    feats = pd.DataFrame([{"symbol": "AAA", "bar_ts": "2026-01-01 10:00:00",
                           **{n: 0.5 for n in FEATURE_NAMES}},
                          {"symbol": "BBB", "bar_ts": "2026-01-02 10:00:00",
                           **{n: 0.5 for n in FEATURE_NAMES}}])
    preds = pd.DataFrame([{"symbol": "AAA", "bar_ts": "2026-01-01 10:00:00",
                           "resolved_at": "2026-01-01 11:00:00", "pwin_side": "+"}])
    fp, pp = tmp_path / "f.csv", tmp_path / "p.csv"
    feats.to_csv(fp, index=False)
    preds.to_csv(pp, index=False)
    out = bld.build_unlabeled(feats_path=fp, pred_path=pp)
    assert list(out["symbol"]) == ["BBB"]
    # `before` NEDENSELLİK kapısıdır: eğitim penceresi dışındaki satır dönmemeli.
    assert bld.build_unlabeled(feats_path=fp, pred_path=pp,
                               before="2026-01-01T00:00:00+00:00").empty


def test_build_unlabeled_eksik_kolon_notr_doldurur(tmp_path):
    """NaN ön-eğitime SIZMAMALI (maskeli rekonstrüksiyon NaN hedefte tanımsızdır)."""
    import pandas as pd

    from nihai.signals.meta_features import FEATURE_NAMES, neutral_default
    from research import build_live_dataset as bld

    feats = pd.DataFrame([{"symbol": "AAA", "bar_ts": "2026-01-01 10:00:00",
                           **{n: (np.nan if n == FEATURE_NAMES[0] else 0.5)
                              for n in FEATURE_NAMES[:-1]}}])
    fp, pp = tmp_path / "f.csv", tmp_path / "p.csv"
    feats.to_csv(fp, index=False)
    pd.DataFrame([{"symbol": "X", "bar_ts": "x", "resolved_at": "", "pwin_side": ""}]).to_csv(
        pp, index=False)
    out = bld.build_unlabeled(feats_path=fp, pred_path=pp)
    assert len(out) == 1
    assert not out[list(FEATURE_NAMES)].isna().any().any()
    assert out[FEATURE_NAMES[0]].iloc[0] == neutral_default(FEATURE_NAMES[0])
    assert out[FEATURE_NAMES[-1]].iloc[0] == neutral_default(FEATURE_NAMES[-1])
