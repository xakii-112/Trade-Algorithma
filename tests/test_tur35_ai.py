"""Tur-35 (AI v7) yetenek testleri — ULTRA basamağı · dağılımsal oynaklık · maskeli
öz-denetimli ön-eğitim · çevrimiçi kaynak füzyonu.

Ağır testler küçük tutulur (mini n, max_epochs<=2); 772M'lik gerçek yapılandırma YALNIZ
`theoretical_param_count` ile BİR KEZ ölçülür (o çağrı lru_cache'lidir).
Torch'lu testler `@TORCH` ile atlanır (modül fail-soft — importorskip DEĞİL).
"""
from __future__ import annotations

import numpy as np
import pytest

from nihai.ai import learning, models, neural

TORCH = pytest.mark.skipif(not neural.TORCH_AVAILABLE, reason="torch yok")


@pytest.fixture()
def feat() -> list[str]:
    return [f"f{i}" for i in range(6)]


def _mini_ultra(names: list[str]):
    """Küçük ULTRA örneği: mimari SÖZLEŞMESİ (ince-taneli uzman + paylaşımlı uzman +
    quantile başlığı) korunur, 772M'lik yapılandırma test süresine girmez."""
    return neural.QuantTransformerUltra(names, d_model=32, n_blocks=2, n_heads=2,
                                        expert_ffn=16, n_routed=4, top_k=2, shared_ffn=16)


# ---------------------------------------------------------------------------
# 1. Kapasite merdiveni — v7 basamağı
# ---------------------------------------------------------------------------
@TORCH
def test_ultra_param_sozlesmesi():
    """v7 SÖZLEŞMESİ: `quant_transformer_ultra` >= 740M ve MoE-XL'in ≈2,7 katı. Taban
    `n_features`'tan BAĞIMSIZDIR (blok yığını tek başına ≈764M) → 8 özellikle ölçmek yeter."""
    p_ultra = neural.theoretical_param_count("quant_transformer_ultra", 8)
    p_moexl = neural.theoretical_param_count("quant_transformer_moe_xl", 8)
    assert p_ultra is not None and p_moexl is not None
    assert p_ultra >= 740_000_000
    assert 2.5 <= p_ultra / p_moexl <= 3.2


@TORCH
def test_merdiven_sirasi_artan():
    """Merdiven SIRASI anlamlıdır (recommend_tier eşikleri onu varsayar): her basamak
    bir öncekinden BÜYÜK olmalı."""
    counts = [neural.theoretical_param_count(t, 8) for t in neural.NEURAL_TIERS]
    assert all(a is not None and b is not None and a < b
               for a, b in zip(counts, counts[1:], strict=False))


def test_recommend_tier_ultra_esigi():
    """Eşik 10M — bugünkü canlı logdan (≈6k satır) DÖRT mertebe uzak ve bu BİLİNÇLİ."""
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")
    assert models.recommend_tier(10_000_000, 96, allow_neural=True) == "quant_transformer_ultra"
    assert models.recommend_tier(9_999_999, 96, allow_neural=True) == "quant_transformer_moe_xl"
    # allow_neural=False → v2 davranışı BİT-DENK (nöral basamak asla önerilmez).
    assert models.recommend_tier(10_000_000, 96) == "mlp"


def test_ultra_param_ladder_raporunda_gorunur():
    rows = models.param_ladder_report(n_features=8, include_neural=True)
    tiers = [r["tier"] for r in rows]
    # Tur-36: merdivenin ucu MEGA'ya taşındı; ULTRA raporda DURUYOR (bu testin iddiası).
    assert "quant_transformer_ultra" in tiers


# ---------------------------------------------------------------------------
# 2. Dağılımsal oynaklık başlığı (`vol_q`)
# ---------------------------------------------------------------------------
@TORCH
def test_ultra_cikti_anahtarlari(feat):
    """ULTRA, MoE-XL'in 8 anahtarını taşır + `vol_q`; diğer basamaklarda `vol_q` YOKTUR."""
    import torch
    m = _mini_ultra(feat).eval()
    with torch.no_grad():
        out = m(torch.zeros(3, len(feat)))
    assert set(out) == {"direction", "magnitude", "vol", "regime", "probe1", "probe2",
                        "moe_balance", "moe_zloss", "vol_q"}
    assert out["vol_q"].shape == (3, len(neural.VOL_QUANTILES))
    xl = neural._build_torch_model("quant_transformer_xl", feat, seed=1).eval()
    with torch.no_grad():
        assert "vol_q" not in xl(torch.zeros(2, len(feat)))


@TORCH
def test_vol_quantile_caprazlamasi_yapisi_geregi_imkansiz(feat):
    """ÇAPRAZ GEÇİŞ imkânsız: sıra kayıpla değil MİMARİYLE (kümülatif softplus) garanti.
    Ağırlıklar kasten savrulsa bile ihlal edilemez.

    İDDİANIN SINIRI: garanti AZALMAYAN'dır, "kesin artan" değil — softplus çok negatif
    girdide float32'de 0.0'a alt-taşar ve iki komşu quantile eşitlenebilir. Test tam olarak
    kurulabilen iddiayı kilitler (Tur-33: iddiayı algoritma belirler, umut değil)."""
    import torch
    torch.manual_seed(5)
    m = _mini_ultra(feat)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 1.5)
    m.eval()
    with torch.no_grad():
        vq = m(torch.randn(64, len(feat)) * 10.0)["vol_q"].numpy()
    assert (np.diff(vq, axis=1) >= 0).all()
    # Normal (savrulmamış) ağırlıklarda demet gerçekten AYRIŞIR — hepsi eşit değildir.
    m2 = _mini_ultra(feat).eval()
    with torch.no_grad():
        vq2 = m2(torch.randn(64, len(feat)))["vol_q"].numpy()
    assert (np.diff(vq2, axis=1) >= 0).all()
    assert float(np.diff(vq2, axis=1).max()) > 0.0


@TORCH
def test_pinball_kaybi_bilinen_deger():
    """Pinball tanımı: e = y − q̂, kayıp = max(τ·e, (τ−1)·e)."""
    import torch
    q = torch.zeros(1, 3)
    taus = (0.1, 0.5, 0.9)
    assert abs(float(neural._pinball_loss(q, torch.tensor([2.0]), taus))
               - (0.1 * 2 + 0.5 * 2 + 0.9 * 2) / 3.0) < 1e-6
    assert abs(float(neural._pinball_loss(q, torch.tensor([-2.0]), taus))
               - (0.9 * 2 + 0.5 * 2 + 0.1 * 2) / 3.0) < 1e-6
    # Kayıp sıfırlanma noktası: tahmin hedefe eşitse kayıp 0.
    assert float(neural._pinball_loss(torch.zeros(1, 3), torch.tensor([0.0]), taus)) == 0.0


@TORCH
def test_ultra_artan_olmayan_quantile_listesi_hata(feat):
    with pytest.raises(ValueError, match="KESİN ARTAN"):
        neural.QuantTransformerUltra(feat, d_model=16, n_blocks=1, n_heads=2, expert_ffn=8,
                                     n_routed=2, shared_ffn=8, vol_quantiles=(0.5, 0.2))


@TORCH
def test_ultra_aciklanabilirligi_bozmaz(feat):
    """Dizi düzeni ([CLS]+3×[SCALE]+özellikler) ve `nn.MultiheadAttention` KORUNDU →
    `explain.attention_rollout` yeni basamakta da çalışmalı. Bu, her yeni basamakta
    bilinçli olarak ödenen bir bedeldir (özel dikkat implementasyonu açıklanabilirliği
    sessizce öldürürdü)."""
    from nihai.ai import explain
    m = _mini_ultra(feat).eval()
    res = explain.attention_rollout(m, np.zeros(len(feat)))
    assert res is not None
    assert res["n_layers"] == 2                      # mini yapılandırmanın blok sayısı
    assert res["seq_len"] == 1 + 3 + len(feat)       # [CLS] + 3×[SCALE] + özellikler
    # Rollout satır-stokastiktir: [CLS] satırı toplamı ≈ 1.
    assert abs(sum(res["cls_attention"].values()) - 1.0) < 1e-4
    assert len(res["top5"]) == 5


@TORCH
def test_ultra_moe_yonlendirme_teshisi_calisir(feat):
    """`moe_routing_stats` yeni basamakta da ölçebilmeli (40 uzman görünmeli)."""
    m = _mini_ultra(feat)
    st = neural.moe_routing_stats(m, np.random.RandomState(0).randn(24, len(feat)))
    assert st is not None
    assert st["n_experts"] == 4 and st["n_blocks"] == 2      # mini yapılandırma
    assert all(b["entropy_norm"] is not None for b in st["blocks"])


@TORCH
def test_predict_vol_quantiles_baslik_yoksa_none():
    """Başlığı olmayan basamakta dürüstçe None — "belirsizlik yok" DEĞİL, "ölçülmüyor"."""
    rs = np.random.RandomState(4)
    names = [f"f{i}" for i in range(5)]
    X, y = rs.randn(60, 5), (rs.rand(60) > 0.5).astype(int)
    m = neural.NeuralCapacity("deep_mlp", names, seed=1).fit(X, y, max_epochs=1, batch_size=32)
    assert m.predict_vol_quantiles(X) is None


# ---------------------------------------------------------------------------
# 3. Maskeli özellik ön-eğitimi
# ---------------------------------------------------------------------------
@TORCH
def test_pretrain_deep_mlp_reddedilir():
    """`deep_mlp` token'sızdır → sessizce atlanmaz, ValueError verir."""
    m = neural.NeuralCapacity("deep_mlp", ["a", "b", "c"], seed=1)
    with pytest.raises(ValueError, match="token tabanlı"):
        m.pretrain(np.random.RandomState(0).randn(30, 3))


@TORCH
def test_pretrain_is_fitted_acmaz_ve_tahmini_reddeder():
    """DÜRÜSTLÜK SÖZLEŞMESİ: etiket görmemiş gövdeden p_win okunamaz."""
    rs = np.random.RandomState(2)
    names = [f"f{i}" for i in range(5)]
    X = rs.randn(80, 5)
    m = neural.NeuralCapacity("quant_transformer", names, seed=1)
    m.pretrain(X, max_epochs=2, batch_size=32)
    assert m.is_pretrained is True and m.is_fitted is False
    with pytest.raises(RuntimeError, match="fit edilmedi"):
        m.predict_proba(X)
    with pytest.raises(RuntimeError, match="fit edilmedi"):
        m.param_count()


@TORCH
def test_pretrain_deterministik_ve_rekonstruksiyonu_dusurur():
    """Aynı seed → aynı ölçümler; ve maskeli rekonstrüksiyon kaybı DÜŞMELİ (kayıp sabit
    kalıyorsa reçete bozuktur — "ön-eğitim koştu" demek "öğrendi" demek değildir)."""
    rs = np.random.RandomState(3)
    names = [f"f{i}" for i in range(6)]
    # Özellikler arası GERÇEK yapı: son üçü ilk üçün doğrusal fonksiyonu (öğrenilebilir).
    base = rs.randn(400, 3)
    X = np.column_stack([base, base @ np.array([[1.0, -0.5, 0.2],
                                                [0.3, 0.8, -0.4],
                                                [-0.2, 0.1, 0.9]])])
    a = neural.NeuralCapacity("quant_transformer", names, seed=11).pretrain(
        X, max_epochs=12, batch_size=64)
    b = neural.NeuralCapacity("quant_transformer", names, seed=11).pretrain(
        X, max_epochs=12, batch_size=64)
    ia, ib = a._pretrain_info, b._pretrain_info
    assert ia["val_recon_mse_first"] == ib["val_recon_mse_first"]
    assert ia["val_recon_mse_best"] == ib["val_recon_mse_best"]
    assert ia["val_recon_mse_best"] < ia["val_recon_mse_first"]
    assert ia["masked_per_row"] == max(1, round(0.25 * 6))


@TORCH
def test_pretrain_warm_start_govdeyi_tasir():
    """`warm_start=True` ön-eğitilmiş gövdeyi KULLANIR; gövde yoksa sessizce sıfırdan kurar
    (ön-eğitim OPSİYONELDİR sözleşmesi)."""
    rs = np.random.RandomState(5)
    names = [f"f{i}" for i in range(5)]
    X, y = rs.randn(120, 5), (rs.rand(120) > 0.5).astype(int)
    m = neural.NeuralCapacity("quant_transformer", names, seed=7).pretrain(
        X, max_epochs=2, batch_size=32)
    m.fit(X, y, max_epochs=1, batch_size=32, warm_start=True)
    r = m.meta()["recipe"]
    assert m.is_fitted is True
    assert r["warm_start"] is True and r["warm_start_used"] is True and r["pretrained"] is True
    assert m.meta()["pretrain"]["n_rows"] == 120

    m2 = neural.NeuralCapacity("quant_transformer", names, seed=7)
    m2.fit(X, y, max_epochs=1, batch_size=32, warm_start=True)
    assert m2.is_fitted and m2.meta()["recipe"]["warm_start_used"] is False


@TORCH
def test_pretrain_bilgisi_artefaktla_tasinir(tmp_path):
    """Ön-eğitim kaydı model kartında YAŞAMALI — aksi hâlde "bu model ön-eğitildi mi?"
    sorusu diskteki artefakttan yanıtlanamaz. Rekonstrüksiyon başlığı ise ATILIR:
    şema sıfırdan eğitilmiş modelle AYNI kalır."""
    rs = np.random.RandomState(8)
    names = [f"f{i}" for i in range(4)]
    X, y = rs.randn(70, 4), (rs.rand(70) > 0.5).astype(int)
    m = neural.NeuralCapacity("quant_transformer", names, seed=2).pretrain(
        X, max_epochs=2, batch_size=32)
    m.fit(X, y, max_epochs=1, batch_size=32, warm_start=True)
    m.save(tmp_path / "kart")
    geri = neural.NeuralCapacity.load(tmp_path / "kart")
    assert geri is not None and geri.is_fitted is True
    assert geri.meta()["pretrain"]["n_rows"] == 70
    assert geri.is_pretrained is True
    # Rekonstrüksiyon başlığı/[MASK] gömmesi state_dict'e SIZMAMIŞ olmalı.
    anahtarlar = set(geri._model.state_dict())
    assert not any("mask_token" in k or k.startswith("net.") for k in anahtarlar)


@TORCH
def test_pretrain_gecersiz_girdi_valueerror():
    m = neural.NeuralCapacity("quant_transformer", ["a", "b", "c"], seed=1)
    rs = np.random.RandomState(1)
    with pytest.raises(ValueError, match="mask_ratio"):
        m.pretrain(rs.randn(20, 3), mask_ratio=0.0)
    with pytest.raises(ValueError, match="feature_names"):
        m.pretrain(rs.randn(20, 5))
    with pytest.raises(ValueError, match="2-B"):
        m.pretrain(rs.randn(20))


@TORCH
def test_ensemble_pretrain_uyeleri_farkli_seedle_hazirlar():
    rs = np.random.RandomState(6)
    names = [f"f{i}" for i in range(4)]
    X, y = rs.randn(90, 4), (rs.rand(90) > 0.5).astype(int)
    ens = neural.NeuralEnsemble("quant_transformer", names, seed=3, n_members=2)
    ens.pretrain(X, max_epochs=2, batch_size=32)
    assert ens.is_fitted is False and len(ens.members) == 2
    assert ens.members[0].seed != ens.members[1].seed
    ens.fit(X, y, max_epochs=1, batch_size=32, warm_start=True)
    assert ens.is_fitted is True
    assert ens.meta()["pretrain"]["n_rows"] == 90
    assert ens.predict_vol_quantiles(X) is None      # quant_transformer'da başlık yok


# ---------------------------------------------------------------------------
# 4. Çevrimiçi kaynak füzyonu (Tur-32 açık kalem #4)
# ---------------------------------------------------------------------------
def test_source_fusion_hizalanmamis_girdi_none():
    assert learning.source_fusion([0.5, 0.6], [0.5], [1, 0]) is None
    assert learning.source_fusion([], [], []) is None


def test_source_fusion_yetersiz_ornek_dururum_yetersiz():
    """n < FUSION_MIN_N → uydurma verdikt YOK."""
    rs = np.random.RandomState(1)
    n = 50
    res = learning.source_fusion(list(rs.rand(n)), list(rs.rand(n)),
                                 list((rs.rand(n) > 0.5).astype(int)))
    assert res is not None and res["durum"] == "YETERSIZ" and res["n"] == n


def test_source_fusion_bilgisiz_kaynaklar_tabani_gecemez():
    """Saf gürültü uzmanları koşan taban oranını GEÇEMEZ — ACI DERS #9'un makine hâli."""
    rs = np.random.RandomState(2)
    n = 800
    y = (rs.rand(n) > 0.5).astype(int)
    res = learning.source_fusion(list(np.clip(rs.rand(n), 0.05, 0.95)),
                                 list(np.clip(rs.rand(n), 0.05, 0.95)), list(y))
    assert res is not None
    assert res["durum"] == "TABANI_GECILEMEDI"
    assert res["experts"] == ["pwin", "neural", "base"]
    assert len(res["final_weights"]) == 3
    assert abs(sum(res["final_weights"]) - 1.0) < 0.01


def test_source_fusion_bilgili_kaynak_tabani_gecer():
    """Uzmanlardan biri GERÇEKTEN bilgi taşıyorsa kapı geçilebilmeli — kapı her şeye
    'hayır' diyen bir kapı olsaydı hiçbir şey ölçmezdi (kapının ayırt-ediciliği)."""
    rs = np.random.RandomState(3)
    n = 900
    y = (rs.rand(n) > 0.5).astype(int)
    bilgili = np.where(y == 1, 0.80, 0.20)                 # güçlü ama kusursuz değil
    gurultu = np.clip(rs.rand(n), 0.05, 0.95)
    res = learning.source_fusion(list(bilgili), list(gurultu), list(y))
    assert res is not None
    assert res["durum"] == "TABANI_GECTI"
    assert res["vs_base"] > 0
    assert res["paired_brier_t"] < learning.FUSION_T_THRESHOLD
    assert res["best_expert"] == "pwin"
    assert res["final_weights"][0] > res["final_weights"][1]   # bilgili uzman ağırlık kazanır


def test_source_fusion_rapor_satiri():
    rs = np.random.RandomState(4)
    n = 400
    y = (rs.rand(n) > 0.5).astype(int)
    st = {"strategies": {"GLOBAL": {"n": n, "post_mean": 0.5, "lo95": 0.4, "hi95": 0.6}},
          "source_fusion": learning.source_fusion(
              list(np.clip(rs.rand(n), 0.05, 0.95)),
              list(np.clip(rs.rand(n), 0.05, 0.95)), list(y))}
    rapor = learning.learning_report(st)
    assert "çevrimiçi füzyon" in rapor
    assert "pişmanlık" in rapor
    assert "TABANI_GEC" in rapor


def test_source_fusion_canli_log_semasindan_uretilir(tmp_path, monkeypatch):
    """`update_from_predictions` hizalı üçlüyü CANLI şemadan çıkarır: kaynak KARŞI tarafı
    söylüyorsa olasılık 1−p ile ortak olaya ("canlı yön kazandı mı") çevrilmelidir."""
    satirlar = ["logged_at,symbol,bar_ts,direction,pwin,pwin_side,neural_pwin,"
                "neural_pwin_side,resolved_at,fwd_ret_pct,hit"]
    rs = np.random.RandomState(9)
    for i in range(300):
        d = "+" if i % 2 == 0 else "-"
        # neural KARŞI tarafı söyler → çeviri (1−p) uygulanmalı
        nside = "-" if d == "+" else "+"
        hit = 1 if rs.rand() > 0.5 else 0
        fwd = (1.0 if hit else -1.0) * (1.0 if d == "+" else -1.0)
        satirlar.append(f"2026-08-16T10:00:{i % 60:02d},AAA,2026-08-16T09:00:00,{d},"
                        f"0.60,{'LONG' if d == '+' else 'SHORT'},0.70,"
                        f"{'LONG' if nside == '+' else 'SHORT'},2026-08-16T11:00:00,{fwd},{hit}")
    log = tmp_path / "predictions_log.csv"
    log.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    monkeypatch.setattr(learning, "PRED_LOG", log)
    monkeypatch.setattr(learning, "LEARNING_STATE", tmp_path / "state.json")
    st = learning.update_from_predictions(tail_n=1000)
    fus = st["source_fusion"]
    assert fus is not None and fus["n"] == 300
    assert fus["durum"] in ("TABANI_GECILEMEDI", "TABANI_GECTI")
    # neural hep 0.70 diyor ama HEP karşı taraf → ortak olayda 0.30'a çevrilmiş olmalı:
    # yani "neural" uzmanının kaybı 0.30'luk sabit tahminci kaybına eşit olmalı.
    y_bar = float(np.mean([1 if s.endswith(",1") else 0 for s in satirlar[1:]]))
    beklenen = -(y_bar * np.log(0.30) + (1 - y_bar) * np.log(0.70))
    assert abs(fus["loss_experts"][1] - beklenen) < 1e-6
