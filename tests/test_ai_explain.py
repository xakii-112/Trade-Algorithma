"""nihai.ai.explain testleri — permütasyon önemi, IG/attention-rollout (torch), split-
conformal kapsam garantisi, açıklama kartı/küresel model raporu şablonları.

Ağ/gerçek dosya YOK (explain.py hiçbir DATA_DIR dosyasına yazmaz — saf hesap + rapor
şablonları). Torch'lu testler `explain.TORCH_AVAILABLE` skipif ile korunur (importorskip
DEĞİL — modül fail-soft, SPEC bölüm 4 deseni). nn.Module test sınıfları torch kuruluyken
tanımlanır (neural.py'nin kendi mimari sınıflarını fail-soft koruma deseninin aynısı).
"""
from __future__ import annotations

import numpy as np
import pytest

from nihai.ai import explain, reasoner

TORCH = pytest.mark.skipif(not explain.TORCH_AVAILABLE, reason="torch yok")

if explain.TORCH_AVAILABLE:
    import torch
    from torch import nn

    class _TinyMLP(nn.Module):
        """Dikkat KATMANI OLMAYAN küçük ağ — attention_rollout'un 'transformer değil'
        dalını test eder; integrated_gradients için de genel amaçlı kullanılır."""

        def __init__(self, n_features: int):
            super().__init__()
            self.lin1 = nn.Linear(n_features, 6)
            self.lin2 = nn.Linear(6, 2)

        def forward(self, x):
            return self.lin2(torch.tanh(self.lin1(x)))

    class _TinyTokenTransformer(nn.Module):
        """Her özelliği kendi token'ı sayan minik transformer (attention_rollout testleri).
        `n_extra` sabit ek token (ör. CLS+SCALE benzeri) eklemek için kullanılır."""

        def __init__(self, n_features: int, d: int = 8, n_extra: int = 0):
            super().__init__()
            self.tok_embed = nn.Linear(1, d)
            self.n_extra = n_extra
            if n_extra:
                self.extra = nn.Parameter(torch.zeros(1, n_extra, d))
            self.attn = nn.MultiheadAttention(d, num_heads=2, batch_first=True)
            self.out = nn.Linear(d, 2)

        def forward(self, x):
            b = x.shape[0]
            tok = self.tok_embed(x.unsqueeze(-1))
            if self.n_extra:
                extra = self.extra.expand(b, -1, -1)
                seq = torch.cat([extra, tok], dim=1)
            else:
                seq = tok
            attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
            pooled = attn_out.mean(dim=1)
            return self.out(pooled)

    class _FakeWrapper:
        """NeuralCapacity sarmalayıcı paritesinin minimal duck-type stand-in'i
        (`._model` + `.feature_names`) — explain.py'nin nihai.ai.neural'a import BAĞI
        OLMADAN bu şekli çözebildiğini doğrular."""

        def __init__(self, model, feature_names):
            self._model = model
            self.feature_names = feature_names


# ---------------------------------------------------------------------------
# permutation_importance — model-bağımsız (predict_proba duck-tipi)
# ---------------------------------------------------------------------------
class _ColumnModel:
    """predict_proba YALNIZ belirtilen sütuna bakar (bilinen-cevaplı permütasyon testi)."""

    def __init__(self, col: int, scale: float = 6.0):
        self.col = col
        self.scale = scale

    def predict_proba(self, X):
        z = self.scale * np.asarray(X)[:, self.col]
        return 1.0 / (1.0 + np.exp(-z))


class _PatlayanModel:
    def predict_proba(self, X):
        raise RuntimeError("kasıtlı patlama")


def _xy_bilinen(n=200, seed=0):
    rng = np.random.default_rng(seed)
    col0 = rng.normal(size=n)
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    X = np.column_stack([col0, noise1, noise2])
    y = (col0 > 0).astype(int)
    return X, y


def test_permutation_importance_bilinen_sutun_en_onemli():
    X, y = _xy_bilinen()
    model = _ColumnModel(col=0)
    out = explain.permutation_importance(model, X, y, seed=42)
    assert out[0]["feature"] == "f0"
    assert out[0]["importance"] > 0.3
    assert out[0]["rank"] == 1


def test_permutation_importance_kullanilmayan_sutunlar_sifir_drop():
    X, y = _xy_bilinen()
    model = _ColumnModel(col=0)
    out = explain.permutation_importance(model, X, y, seed=7)
    by_feat = {d["feature"]: d["importance"] for d in out}
    # model yalnız f0'a bakıyor → f1/f2'yi karıştırmak predict_proba'yı HİÇ değiştirmez
    assert by_feat["f1"] == 0.0
    assert by_feat["f2"] == 0.0


def test_permutation_importance_feature_names_kullanilir():
    X, y = _xy_bilinen()
    model = _ColumnModel(col=0)
    out = explain.permutation_importance(model, X, y, feature_names=["ret_20", "a", "b"])
    assert out[0]["feature"] == "ret_20"


def test_permutation_importance_feature_names_uzunluk_uyumsuzsa_generic():
    X, y = _xy_bilinen()
    model = _ColumnModel(col=0)
    out = explain.permutation_importance(model, X, y, feature_names=["tek_isim"])
    assert {d["feature"] for d in out} == {"f0", "f1", "f2"}


def test_permutation_importance_x_y_uzunluk_uyumsuz_bos_liste():
    X, y = _xy_bilinen()
    assert explain.permutation_importance(_ColumnModel(0), X, y[:-5]) == []


def test_permutation_importance_tek_sinif_y_bos_liste():
    X, _ = _xy_bilinen()
    y_tek = np.zeros(len(X), dtype=int)
    assert explain.permutation_importance(_ColumnModel(0), X, y_tek) == []


def test_permutation_importance_x_2b_degilse_bos_liste():
    assert explain.permutation_importance(_ColumnModel(0), [1.0, 2.0, 3.0], [0, 1, 0]) == []


def test_permutation_importance_deterministik_ayni_seed():
    X, y = _xy_bilinen()
    model = _ColumnModel(col=0)
    out1 = explain.permutation_importance(model, X.copy(), y, seed=99)
    out2 = explain.permutation_importance(model, X.copy(), y, seed=99)
    assert out1 == out2


def test_permutation_importance_x_bozulmadan_geri_kalir():
    """Sütun-permütasyonu geçici — çağıranın X dizisi çağrı sonrası DEĞİŞMEMELİ."""
    X, y = _xy_bilinen()
    X_kopya = X.copy()
    explain.permutation_importance(_ColumnModel(0), X, y, seed=3)
    np.testing.assert_array_equal(X, X_kopya)


def test_permutation_importance_model_patlarsa_bos_liste():
    X, y = _xy_bilinen()
    assert explain.permutation_importance(_PatlayanModel(), X, y) == []


# ---------------------------------------------------------------------------
# conformal_calibrate / conformal_interval
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_conformal_calibrate_gecersiz_alpha_none(alpha):
    assert explain.conformal_calibrate([0.5, 0.6], [1, 0], alpha=alpha) is None


def test_conformal_calibrate_bos_veri_none():
    assert explain.conformal_calibrate([], []) is None


def test_conformal_calibrate_uzunluk_uyumsuz_none():
    assert explain.conformal_calibrate([0.5, 0.6, 0.7], [1, 0]) is None


def test_conformal_calibrate_hicgecerli_cift_yoksa_none():
    assert explain.conformal_calibrate(["a", "b"], [1, 0]) is None


def test_conformal_calibrate_k_asim_durumunda_q_hat_bir():
    """n=5, alpha=0.1 → k=ceil(6*0.9)=6 > n=5 → q_hat=1.0 (sınır davranışı, belgeli)."""
    calib = explain.conformal_calibrate([0.5] * 5, [1, 1, 1, 1, 1], alpha=0.1)
    assert calib is not None
    assert calib["q_hat"] == 1.0
    assert calib["n"] == 5


def test_conformal_calibrate_q_hat_0_1_araliginda():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, 500)
    y = (rng.uniform(size=500) < p).astype(int)
    calib = explain.conformal_calibrate(p.tolist(), y.tolist(), alpha=0.1)
    assert calib is not None
    assert 0.0 <= calib["q_hat"] <= 1.0


def test_conformal_calibrate_alpha_kucultukce_q_hat_artar_monoton():
    """Daha düşük alpha = daha yüksek kapsam hedefi → q_hat AZALMAZ (geniş aralık)."""
    rng = np.random.default_rng(11)
    p = rng.uniform(0.05, 0.95, 800)
    y = (rng.uniform(size=800) < p).astype(int)
    q_genis = explain.conformal_calibrate(p.tolist(), y.tolist(), alpha=0.05)["q_hat"]
    q_dar = explain.conformal_calibrate(p.tolist(), y.tolist(), alpha=0.5)["q_hat"]
    assert q_genis >= q_dar


def test_conformal_interval_calib_none():
    assert explain.conformal_interval(0.5, None) is None


def test_conformal_interval_calib_dict_degil():
    assert explain.conformal_interval(0.5, [0.1, 0.2]) is None


def test_conformal_interval_q_hat_eksik():
    assert explain.conformal_interval(0.5, {"method": "x"}) is None


def test_conformal_interval_p_sayisal_degil():
    assert explain.conformal_interval("abc", {"q_hat": 0.1}) is None


def test_conformal_interval_p_nan():
    assert explain.conformal_interval(float("nan"), {"q_hat": 0.1}) is None


def test_conformal_interval_temel_hesap():
    assert explain.conformal_interval(0.5, {"q_hat": 0.2}) == (0.3, 0.7)


def test_conformal_interval_ust_sinir_kirpma():
    lo, hi = explain.conformal_interval(0.95, {"q_hat": 0.9})
    assert hi == 1.0
    assert lo == pytest.approx(0.05)


def test_conformal_interval_alt_sinir_kirpma():
    lo, hi = explain.conformal_interval(0.05, {"q_hat": 0.9})
    assert lo == 0.0
    assert hi == pytest.approx(0.95)


def test_conformal_kapsam_simulasyonu_nominal_90():
    """SPEC ZORUNLU: sentetik kalibrasyon+test kümesinde split-conformal aralığın GERÇEK
    kapsamı nominal (1-alpha)=%90 civarında olmalı. Kalibrasyon/test AYNI (iyi-kalibre
    p → Bernoulli(p) etiket) dağılımdan i.i.d. çekilir (split-conformal'ın exchangeability
    varsayımı) — split-conformal'ın marjinal garantisi P(y ∈ [lo,hi]) >= 1-alpha'dır;
    [0,1] kırpması yalnızca aralığı SIKILAŞTIRABİLİR yönde etki eder (y zaten [0,1] içinde
    olduğundan kırpma kapsamı ASLA düşürmez) → gerçek kapsam nominalin HAFİFÇE üstünde de
    çıkabilir. Tolerans (n=3000+3000, seed sabit): [%85, %100] — belgeli, deterministik."""
    rng = np.random.default_rng(123)
    n_cal, n_test = 3000, 3000
    p_cal = rng.uniform(0.02, 0.98, n_cal)
    y_cal = (rng.uniform(size=n_cal) < p_cal).astype(int)
    calib = explain.conformal_calibrate(p_cal.tolist(), y_cal.tolist(), alpha=0.1)
    assert calib is not None

    p_test = rng.uniform(0.02, 0.98, n_test)
    y_test = (rng.uniform(size=n_test) < p_test).astype(int)
    covered = 0
    for p, y in zip(p_test, y_test, strict=True):
        lo, hi = explain.conformal_interval(float(p), calib)
        if lo <= y <= hi:
            covered += 1
    coverage = covered / n_test
    assert 0.85 <= coverage <= 1.0, f"kapsam %{coverage * 100:.1f} nominal %90 civarında değil"


# ---------------------------------------------------------------------------
# integrated_gradients (torch)
# ---------------------------------------------------------------------------
@TORCH
def test_ig_tamlik_aksiyomu():
    """SPEC ZORUNLU: IG tamlık aksiyomu — Σ atıf ≈ f(x_row) − f(baseline). Tolerans 1e-3
    (n_steps=100 trapez yaklaşıklığı için belgeli; küçük ağ + tanh düzgün/sınırlı gradyanlı)."""
    torch.manual_seed(0)
    model = _TinyMLP(4)
    model.eval()
    x_row = [0.3, -0.2, 0.5, 0.1]
    out = explain.integrated_gradients(model, x_row, steps=100)
    assert out is not None
    assert out["completeness_delta"] < 1e-3


@TORCH
def test_ig_baseline_none_sifir_vektor_varsayilan():
    torch.manual_seed(1)
    model = _TinyMLP(3)
    model.eval()
    out_none = explain.integrated_gradients(model, [1.0, -1.0, 0.5], baseline=None, steps=30)
    out_sifir = explain.integrated_gradients(model, [1.0, -1.0, 0.5], baseline=[0.0, 0.0, 0.0], steps=30)
    assert out_none is not None and out_sifir is not None
    assert out_none["attributions"] == out_sifir["attributions"]


@TORCH
def test_ig_deterministik():
    torch.manual_seed(2)
    model = _TinyMLP(3)
    model.eval()
    x_row = [0.2, 0.4, -0.3]
    out1 = explain.integrated_gradients(model, x_row, steps=40)
    out2 = explain.integrated_gradients(model, x_row, steps=40)
    assert out1 == out2


@TORCH
def test_ig_feature_names_wrapper_duck_type():
    torch.manual_seed(3)
    model = _TinyMLP(3)
    model.eval()
    wrapper = _FakeWrapper(model, feature_names=["ret_5", "rsi", "vol_20"])
    out = explain.integrated_gradients(wrapper, [0.1, 0.2, 0.3], steps=20)
    assert out is not None
    assert set(out["feature_names"]) == {"ret_5", "rsi", "vol_20"}
    assert len(out["top5"]) <= 3


@TORCH
def test_ig_bozuk_x_row_none():
    model = _TinyMLP(3)
    model.eval()
    assert explain.integrated_gradients(model, ["a", "b", "c"]) is None
    assert explain.integrated_gradients(model, []) is None
    assert explain.integrated_gradients(model, [1.0, float("nan"), 2.0]) is None


@TORCH
def test_ig_baseline_sekil_uyumsuz_none():
    model = _TinyMLP(3)
    model.eval()
    assert explain.integrated_gradients(model, [1.0, 2.0, 3.0], baseline=[0.0, 0.0]) is None


def test_ig_torch_yokken_none(monkeypatch):
    monkeypatch.setattr(explain, "TORCH_AVAILABLE", False)
    assert explain.integrated_gradients(object(), [1.0, 2.0]) is None


# ---------------------------------------------------------------------------
# attention_rollout (torch)
# ---------------------------------------------------------------------------
@TORCH
def test_attention_rollout_transformer_olmayan_model_none():
    model = _TinyMLP(4)
    model.eval()
    assert explain.attention_rollout(model, [0.1, 0.2, 0.3, 0.4]) is None


@TORCH
def test_attention_rollout_calisir_ve_satir_stokastik():
    torch.manual_seed(4)
    model = _TinyTokenTransformer(n_features=3, n_extra=0)
    model.eval()
    out = explain.attention_rollout(model, [0.1, -0.2, 0.3])
    assert out is not None
    assert out["seq_len"] == 3
    assert out["n_layers"] == 1
    toplam = sum(out["cls_attention"].values())
    assert toplam == pytest.approx(1.0, abs=1e-4)   # rollout satır-stokastik (test kanıtı)


@TORCH
def test_attention_rollout_feature_names_ile_etiketleme():
    torch.manual_seed(5)
    inner = _TinyTokenTransformer(n_features=3, n_extra=4)   # CLS + 3 SCALE benzeri + 3 özellik
    inner.eval()
    wrapper = _FakeWrapper(inner, feature_names=["ret_5", "rsi", "vol_20"])
    out = explain.attention_rollout(wrapper, [0.1, 0.2, 0.3])
    assert out is not None
    assert out["seq_len"] == 7
    etiketler = set(out["cls_attention"].keys())
    assert {"CLS", "ret_5", "rsi", "vol_20"} <= etiketler
    assert sum(out["cls_attention"].values()) == pytest.approx(1.0, abs=1e-4)


@TORCH
def test_attention_rollout_hook_temizligi_tekrar_cagirilabilir():
    """Hook'lar finally'de kaldırılmazsa 2. çağrıda katman/ağırlık sayısı KATLANIRDI —
    bu regresyon testi hook temizliğini dolaylı doğrular."""
    torch.manual_seed(6)
    model = _TinyTokenTransformer(n_features=2, n_extra=0)
    model.eval()
    x = [0.5, -0.5]
    out1 = explain.attention_rollout(model, x)
    out2 = explain.attention_rollout(model, x)
    assert out1["n_layers"] == out2["n_layers"] == 1
    assert out1 == out2   # deterministik (eval modunda, dropout yok)


@TORCH
def test_attention_rollout_bozuk_x_row_none():
    model = _TinyTokenTransformer(n_features=3)
    model.eval()
    assert explain.attention_rollout(model, ["a", "b", "c"]) is None
    assert explain.attention_rollout(model, []) is None


def test_attention_rollout_torch_yokken_none(monkeypatch):
    monkeypatch.setattr(explain, "TORCH_AVAILABLE", False)
    assert explain.attention_rollout(object(), [1.0, 2.0]) is None


# ---------------------------------------------------------------------------
# decision_card
# ---------------------------------------------------------------------------
def _ornek_decision_dict():
    return {"symbol": "AAPL", "strategy": "event_breakout", "action": "LONG_BIAS",
            "posterior": 0.62, "conviction": 0.24, "kelly_suggest": 0.01, "utility": 0.15}


def test_decision_card_temel_alanlar_ve_closing():
    card = explain.decision_card(_ornek_decision_dict())
    assert card["card"]["symbol"] == "AAPL"
    assert card["card"]["action"] == "LONG_BIAS"
    assert reasoner.CLOSING in card["text"]


def test_decision_card_attributions_list_girdi():
    attrs = [{"feature": "ret_20", "importance": 0.05, "rank": 1},
             {"feature": "rsi", "importance": 0.02, "rank": 2}]
    card = explain.decision_card(_ornek_decision_dict(), attributions=attrs)
    assert card["card"]["top_attributions"][0]["feature"] == "ret_20"
    assert "ret_20" in card["text"]


def test_decision_card_attributions_dict_girdi():
    attrs = {"ret_20": 0.05, "rsi": -0.2, "vol_20": 0.001}
    card = explain.decision_card(_ornek_decision_dict(), attributions=attrs)
    assert card["card"]["top_attributions"][0]["feature"] == "rsi"   # |-0.2| en büyük


def test_decision_card_conformal_kapsiyor_belirsiz():
    card = explain.decision_card(_ornek_decision_dict(), conformal=(0.4, 0.6))
    assert card["card"]["conformal_straddles_half"] is True
    assert "kararsız" in card["text"]


def test_decision_card_conformal_kapsamiyor_destekli():
    card = explain.decision_card(_ornek_decision_dict(), conformal=(0.55, 0.7))
    assert card["card"]["conformal_straddles_half"] is False
    assert "destekli" in card["text"]


def test_decision_card_conformal_yok_pasif_not():
    card = explain.decision_card(_ornek_decision_dict())
    assert card["card"]["conformal_straddles_half"] is None
    assert "pasif" in card["text"]


def test_decision_card_eksik_alanlar_cokmez():
    card = explain.decision_card({})
    assert card["card"]["symbol"] == "?"
    assert reasoner.CLOSING in card["text"]


def test_decision_card_decision_dict_none_cokmez():
    card = explain.decision_card(None)  # type: ignore[arg-type]
    assert card["card"]["action"] == "?"


def test_decision_card_extras_metne_yansir():
    card = explain.decision_card(_ornek_decision_dict(), extras={"opex_week": True})
    assert card["card"]["extras"] == {"opex_week": True}
    assert "opex_week" in card["text"]


# ---------------------------------------------------------------------------
# global_model_report
# ---------------------------------------------------------------------------
def test_global_model_report_temel_yapi_ve_closing():
    X, y = _xy_bilinen(n=100)
    model = _ColumnModel(col=0)
    text = explain.global_model_report(model, X, y)
    assert "Permütasyon önemi" in text
    assert "Kısmi bağımlılık" in text
    assert text.endswith(reasoner.CLOSING)


def test_global_model_report_tek_sinif_hesaplanamadi_notu():
    X, _ = _xy_bilinen(n=50)
    y_tek = np.zeros(50, dtype=int)
    text = explain.global_model_report(_ColumnModel(0), X, y_tek)
    assert "hesaplanamadı" in text
    assert text.endswith(reasoner.CLOSING)


def test_global_model_report_kucuk_x_pdp_hesaplanamadi():
    X = np.array([[1.0, 2.0]])   # tek satır → PDP/permütasyon anlamsız
    y = np.array([1])
    text = explain.global_model_report(_ColumnModel(0), X, y)
    assert text.endswith(reasoner.CLOSING)


def test_global_model_report_feature_names_kullanilir():
    X, y = _xy_bilinen(n=80)
    text = explain.global_model_report(_ColumnModel(0), X, y, feature_names=["a", "b", "c"])
    assert " a:" in text or "  a:" in text
