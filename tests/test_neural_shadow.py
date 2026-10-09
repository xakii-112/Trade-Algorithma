"""test_neural_shadow.py — nöral GÖLGE p_win köprüsü (Tur-18).

Kapsam: model seçim kilitleri (sentetik/şema/artefakt), fail-soft sözleşmesi,
`is_long`in aday tarafına göre kurulması, CSV şema göçü + kolon yazımı ve
learning.py'nin beklediği kolon adlarının SÖZLEŞME kilidi.

Hiçbiri torch/gerçek model gerektirmez (stub model); gerçek çıkarım ayrı bir
duman testinin konusudur.
"""
from __future__ import annotations

import csv

import pytest

import nihai.execution.live_predict_service as lps
import nihai.execution.neural_shadow as ns
import nihai.execution.prediction_tracker as pt
from nihai.signals.meta_features import FEATURE_NAMES
from nihai.signals.strategy import Signal, SignalAction


# --------------------------------------------------------------------------- stub
class _StubModel:
    """predict_with_uncertainty'yi taklit eder; verilen X'i saklar (hiza denetimi)."""

    def __init__(self, tier: str = "deep_mlp") -> None:
        self.tier = tier
        self.feature_names = list(FEATURE_NAMES)
        self.last_x: list[list[float]] | None = None
        self.last_T: int | None = None

    def predict_with_uncertainty(self, X, T: int = 20):
        self.last_x = [list(map(float, r)) for r in X]
        self.last_T = T
        # is_long'a BAĞLI çıktı: taraf gerçekten vektöre yansıyor mu, test görebilsin.
        i = self.feature_names.index("is_long")
        p = 0.7 if self.last_x[0][i] >= 0.5 else 0.3
        return [p], [0.05]


@pytest.fixture()
def stub_shadow(monkeypatch):
    """Yüklü modeli stub'la değiştirilmiş, AKTİF bir gölge çıkarıcı."""
    monkeypatch.delenv("NEURAL_SHADOW", raising=False)
    monkeypatch.setenv("NEURAL_SHADOW_MODEL", "")
    sh = ns._NeuralShadow.__new__(ns._NeuralShadow)   # _load() çalıştırmadan kur
    sh.tier, sh.card_hash, sh.reason = "deep_mlp", "abc123", None
    sh._model = _StubModel()
    sh._features = list(FEATURE_NAMES)
    sh._passes = 7
    return sh


def _feats(**over) -> dict[str, float]:
    f = {n: 0.0 for n in FEATURE_NAMES}
    f.update(over)
    return f


# --------------------------------------------------------------------- kapatma/fail-soft
def test_env_off_pasif_ve_neden_gorunur(monkeypatch):
    monkeypatch.setenv("NEURAL_SHADOW", "off")
    sh = ns._NeuralShadow()
    assert sh.active is False
    assert sh.reason == "NEURAL_SHADOW=off"
    assert sh.pwin(_feats(), "LONG") is None
    assert sh.status()["active"] is False


def test_gecersiz_taraf_ve_bos_feats_none(stub_shadow):
    assert stub_shadow.pwin(_feats(), "WAIT") is None
    assert stub_shadow.pwin(_feats(), "") is None
    assert stub_shadow.pwin({}, "LONG") is None


# --------------------------------------------------------------- v4 (Tur-22): OOD eşiği
def test_max_z_auto_modu_env_den_okunur(monkeypatch):
    monkeypatch.setenv("NEURAL_SHADOW", "off")   # yükleme yapılmasın, yalnız env ayrıştırma
    monkeypatch.setenv("NEURAL_SHADOW_MAX_Z", "auto")
    sh = ns._NeuralShadow()
    assert sh._max_z_mode == "auto"
    assert sh._max_z == 6.0        # model yüklenmediği için henüz sabit tabanda


def test_max_z_sayisal_ve_bozuk_deger_fixed(monkeypatch):
    monkeypatch.setenv("NEURAL_SHADOW", "off")
    monkeypatch.setenv("NEURAL_SHADOW_MAX_Z", "3.5")
    assert ns._NeuralShadow()._max_z == 3.5
    monkeypatch.setenv("NEURAL_SHADOW_MAX_Z", "abc")
    sh = ns._NeuralShadow()
    assert sh._max_z == 6.0 and sh._max_z_mode == "fixed"


def test_status_v4_alanlari(stub_shadow):
    st = stub_shadow.status()
    for key in ("ensemble", "max_z_mode", "train_zmax_q99"):
        assert key in st
    assert st["ensemble"] == 1     # sınıf-düzeyi varsayılan (stub __init__'i atladı)


# ------------------------------------------------- v4: topluluk artefaktı canlı köprüde
def test_topluluk_artefakti_koprude_yuklenir_ve_uretir(tmp_path, monkeypatch):
    """GERÇEK entegrasyon: `NeuralEnsemble` diske yazılır → köprü `load_any` ile onu
    yükler → p_win üretir. Topluluk desteğinin canlı yolda GERÇEKTEN çalıştığının kanıtı
    (birim testleri yalnız neural.py tarafını gösterirdi)."""
    import numpy as np

    from nihai.ai import neural
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(float)
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=list(FEATURE_NAMES), seed=1,
                                n_members=2).fit(X, y, max_epochs=1)
    base = tmp_path / "topluluk"
    ens.save(base)

    monkeypatch.delenv("NEURAL_SHADOW", raising=False)
    monkeypatch.setenv("NEURAL_SHADOW_MODEL", str(base))
    monkeypatch.setenv("NEURAL_SHADOW_MAX_Z", "auto")
    sh = ns._NeuralShadow()
    assert sh.active is True, sh.reason
    assert sh.n_members == 2
    st = sh.status()
    assert st["ensemble"] == 2
    assert st["max_z_mode"] == "auto"
    assert st["max_z"] >= 4.0            # veri-türetilmiş eşik tabanı
    res = sh.pwin(_feats(), "LONG")
    assert res is not None and 0.0 <= res[0] <= 1.0 and res[1] >= 0.0


def test_auto_esik_q99_yoksa_sabite_duser(tmp_path, monkeypatch):
    """Eski artefaktta `train_zmax_q99` yok → auto istense bile sabit 6,0'a düşer
    (sessiz yanlış eşik yerine dürüst degrade)."""
    import json

    import numpy as np

    from nihai.ai import neural
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, len(FEATURE_NAMES)))
    m = neural.NeuralCapacity("deep_mlp", feature_names=list(FEATURE_NAMES), seed=1).fit(
        X, (X[:, 0] > 0).astype(float), max_epochs=1)
    base = tmp_path / "eski"
    m.save(base)
    payload = json.loads((tmp_path / "eski.json").read_text(encoding="utf-8"))
    payload.pop("train_zmax_q99", None)
    (tmp_path / "eski.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.delenv("NEURAL_SHADOW", raising=False)
    monkeypatch.setenv("NEURAL_SHADOW_MODEL", str(base))
    monkeypatch.setenv("NEURAL_SHADOW_MAX_Z", "auto")
    sh = ns._NeuralShadow()
    assert sh.active is True, sh.reason
    assert sh._max_z == 6.0
    assert sh.status()["train_zmax_q99"] is None


def test_cikarim_hatasi_yutulur(stub_shadow, monkeypatch):
    """Model patlarsa kayıt akışı DURMAZ (tahmin satırı gölgeden önemlidir)."""
    def boom(*_a, **_k):
        raise RuntimeError("torch patladı")
    monkeypatch.setattr(stub_shadow._model, "predict_with_uncertainty", boom)
    assert stub_shadow.pwin(_feats(), "LONG") is None


@pytest.mark.parametrize(("p", "sd", "beklenen"), [
    (0.0, 0.0, None),            # doygun aşağı — dağılım-dışı artefakt
    (1.0, 0.0, None),            # doygun yukarı
    (0.0, 0.02, (0.0, 0.02)),    # uç ama varyanslı → gerçek tahmin, kaydedilir
    (0.36, 0.0, (0.36, 0.0)),    # varyansız ama uç DEĞİL → dejenere model, ölçülmeli
])
def test_doygunluk_guardi(stub_shadow, monkeypatch, p, sd, beklenen):
    """Canlı duman testinde çıktı: ısınmamış/dağılım-dışı vektör ağı doyurup p=0.0,
    std=0.0 üretiyor. Sahte-kesinlik Brier'i kirletmemeli."""
    monkeypatch.setattr(stub_shadow._model, "predict_with_uncertainty",
                        lambda X, T=20: ([p], [sd]))
    assert stub_shadow.pwin(_feats(), "LONG") == beklenen


def test_env_model_yoksa_pasif(monkeypatch, tmp_path):
    monkeypatch.delenv("NEURAL_SHADOW", raising=False)
    monkeypatch.setenv("NEURAL_SHADOW_MODEL", str(tmp_path / "yok"))
    sh = ns._NeuralShadow()
    assert sh.active is False
    assert "artefakt" in (sh.reason or "")


# ------------------------------------------------------------------ taraf → is_long
def test_is_long_aday_tarafina_gore_kurulur(stub_shadow):
    """Aday LONG iken is_long=1 olmalı: aksi halde sessizce SHORT'un olasılığı döner."""
    i = FEATURE_NAMES.index("is_long")
    p_long, std = stub_shadow.pwin(_feats(is_long=0.0), "LONG")   # yanlış is_long ile gelse bile
    assert stub_shadow._model.last_x[0][i] == 1.0
    assert (p_long, std) == (0.7, 0.05)
    p_short, _ = stub_shadow.pwin(_feats(is_long=1.0), "SHORT")
    assert stub_shadow._model.last_x[0][i] == 0.0
    assert p_short == 0.3
    assert stub_shadow._model.last_T == 7        # MC geçiş sayısı iletiliyor


def test_vektor_modelin_kendi_sirasinda_kurulur(stub_shadow):
    """Model kendi feature_names sırasını dayatır (train/serve hizası)."""
    stub_shadow._features = list(reversed(FEATURE_NAMES))
    stub_shadow._model.feature_names = list(reversed(FEATURE_NAMES))
    stub_shadow.pwin(_feats(adx14=42.0), "LONG")
    j = list(reversed(FEATURE_NAMES)).index("adx14")
    assert stub_shadow._model.last_x[0][j] == 42.0


# ------------------------------------------------------------------- model seçim kilitleri
def _card(**over):
    c = {"kind": "neural", "feature_names": list(FEATURE_NAMES),
         "data": {"source": "data/meta_dataset_pit.csv"},
         "artifacts": {"base_path": "data/ai_models_artifacts/x"}}
    c.update(over)
    return c


def test_eligible_kilitleri():
    assert ns._eligible(_card()) is True
    assert ns._eligible(_card(kind="meta")) is False                        # nöral değil
    assert ns._eligible(_card(data={"source": "synthetic:4000"})) is False  # sentetik canlıya girmez
    assert ns._eligible(_card(feature_names=["adx14"])) is False            # şema uyuşmuyor
    assert ns._eligible(_card(artifacts={})) is False                       # artefakt yok


def test_sentetik_kart_secilmez(monkeypatch, tmp_path):
    """Sentetik müfredat modeli DAHA YENİ olsa bile canlı bara sokulmaz."""
    monkeypatch.delenv("NEURAL_SHADOW", raising=False)
    monkeypatch.delenv("NEURAL_SHADOW_MODEL", raising=False)
    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    cards = [_card(data={"source": "synthetic:4000"}, registered_at="2099-01-01T00:00:00+00:00")]
    monkeypatch.setattr(ns, "_resolve_base", lambda _p: tmp_path / "x")

    import nihai.ai.registry as registry
    monkeypatch.setattr(registry, "list_models", lambda *_a, **_k: cards)
    sh = ns._NeuralShadow.__new__(ns._NeuralShadow)
    sh.reason = None
    base, card = sh._pick()
    assert base is None and card is None
    assert "uygun nöral model kartı yok" in (sh.reason or "")


def test_resolve_base_data_koku_data_dir_e_baglanir(tmp_path, monkeypatch):
    """`data/...` yolları NIHAI_DATA_DIR yönlendirmesine saygı duymalı (test izolasyonu)."""
    monkeypatch.setattr(ns, "DATA_DIR", tmp_path)
    assert ns._resolve_base("data/ai_models_artifacts/m") == tmp_path / "ai_models_artifacts" / "m"
    assert ns._resolve_base(str(tmp_path / "abs")) == tmp_path / "abs"
    assert ns._resolve_base("docs/m") == ns.ROOT / "docs" / "m"


# ----------------------------------------------------------------- CSV hücreleri + şema
def test_neural_cells_bicimleri():
    assert pt._neural_cells({"_neural": {"pwin": 0.61234, "std": 0.0512, "side": "long"}}) \
        == ("0.6123", "LONG", "0.0512")
    assert pt._neural_cells({}) == ("", "", "")
    assert pt._neural_cells({"_neural": {"pwin": 0.5, "side": "WAIT"}}) == ("", "", "")
    assert pt._neural_cells({"_neural": {"pwin": "bozuk", "side": "LONG"}}) == ("", "", "")
    # std bozuk/eksik olabilir ama p_win yazılmaya devam eder
    assert pt._neural_cells({"_neural": {"pwin": 0.4, "side": "SHORT"}}) == ("0.4", "SHORT", "")


def test_fields_learning_sozlesmesi():
    """learning.py bu İKİ adı arıyor; yeniden adlandırma köprüyü sessizce öldürür."""
    assert "neural_pwin" in pt.FIELDS
    assert "neural_pwin_side" in pt.FIELDS
    assert "neural_pwin_std" in pt.FIELDS


def test_record_neural_kolonlarini_yazar(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "p.csv")
    monkeypatch.setattr(pt, "FEATS_LOG", tmp_path / "f.csv")
    rows = [
        {"symbol": "AAPL", "action": "LONG", "direction": "+", "price": 100.0,
         "bar_ts": "2026-07-25 14:00:00", "pwin": 0.55, "pwin_side": "LONG",
         "_neural": {"pwin": 0.42, "std": 0.03, "side": "LONG"}},
        {"symbol": "MSFT", "action": "SHORT", "direction": "-", "price": 50.0,
         "bar_ts": "2026-07-25 14:00:00", "pwin": 0.51, "pwin_side": "SHORT"},   # gölge yok
    ]
    assert pt.record(rows, "1h") == 2
    out = {r["symbol"]: r for r in csv.DictReader(pt.PRED_LOG.open(encoding="utf-8"))}
    assert out["AAPL"]["neural_pwin"] == "0.42"
    assert out["AAPL"]["neural_pwin_side"] == "LONG"
    assert out["AAPL"]["neural_pwin_std"] == "0.03"
    assert out["MSFT"]["neural_pwin"] == ""          # gölge pasifken boş kolon
    assert out["MSFT"]["pwin"] == "0.51"             # mevcut kanıt döngüsü bozulmadı


def test_eski_semali_log_gocu(tmp_path, monkeypatch):
    """Kolon eklenince eski dosya atomik yükseltilir; eski satırlar '' alır, veri kaymaz."""
    log = tmp_path / "p.csv"
    monkeypatch.setattr(pt, "PRED_LOG", log)
    monkeypatch.setattr(pt, "FEATS_LOG", tmp_path / "f.csv")
    eski = [f for f in pt.FIELDS if not f.startswith("neural_")]
    with log.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eski)
        w.writeheader()
        w.writerow({**{k: "" for k in eski}, "symbol": "NVDA", "bar_ts": "2026-07-24 10:00:00",
                    "direction": "+", "action": "LONG", "price": "9", "pwin": "0.6",
                    "pwin_side": "LONG", "hit": "1", "resolved_at": "2026-07-24 15:00:00"})
    pt.record([{"symbol": "AAPL", "action": "LONG", "direction": "+", "price": 100.0,
                "bar_ts": "2026-07-25 14:00:00", "pwin": None, "pwin_side": "LONG",
                "_neural": {"pwin": 0.33, "std": 0.02, "side": "LONG"}}], "1h")
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    old = next(r for r in rows if r["symbol"] == "NVDA")
    assert old["neural_pwin"] == "" and old["hit"] == "1" and old["pwin"] == "0.6"
    new = next(r for r in rows if r["symbol"] == "AAPL")
    assert new["neural_pwin"] == "0.33"


def test_learning_neural_brier_i_uctan_uca_doldurur(tmp_path, monkeypatch):
    """record() → learning.py: köprünün iki ucu gerçekten birbirine bağlı mı."""
    import nihai.ai.learning as learning
    log = tmp_path / "p.csv"
    monkeypatch.setattr(pt, "PRED_LOG", log)
    monkeypatch.setattr(pt, "FEATS_LOG", tmp_path / "f.csv")
    monkeypatch.setattr(learning, "PRED_LOG", log)
    monkeypatch.setattr(learning, "LEARNING_STATE", tmp_path / "ai_learning_state.json")
    rows = [{"symbol": f"S{i}", "action": "LONG", "direction": "+", "price": 10.0,
             "bar_ts": f"2026-07-25 1{i}:00:00", "pwin": 0.5, "pwin_side": "LONG",
             "_neural": {"pwin": 0.8, "std": 0.01, "side": "LONG"}} for i in range(4)]
    assert pt.record(rows, "1h") == 4
    # Çöz: kazanan yön (+) → gölge taraf LONG da kazandı.
    text = log.read_text(encoding="utf-8").splitlines()
    head = text[0].split(",")
    ri, hi, fi = head.index("resolved_at"), head.index("hit"), head.index("fwd_ret_pct")
    fixed = [text[0]]
    for line in text[1:]:
        c = line.split(",")
        c[ri], c[hi], c[fi] = "2026-07-25 20:00:00", "1", "1.5"
        fixed.append(",".join(c))
    log.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    state = learning.update_from_predictions()
    assert state["sources"]["neural"]["n"] == 4
    assert state["sources"]["neural"]["brier"] == pytest.approx(0.04, abs=1e-6)   # (1-0.8)^2


# ------------------------------------------------------------- canlı servis köprüsü
def _sig(action, **md):
    return Signal(action, "AAPL", 100.0, 95.0, 110.0, reasons=[], quality_score=1.0,
                  rr_ratio=2.0, metadata=dict(md))


def test_candidate_side_tek_kaynak():
    assert lps._candidate_side(_sig(SignalAction.LONG)) == "LONG"
    assert lps._candidate_side(_sig(SignalAction.SHORT)) == "SHORT"
    # enforce kipi: kapı WAIT'e çevirdi ama elenen yön metadata'da
    assert lps._candidate_side(_sig(SignalAction.WAIT, meta_p_win=0.3, meta_gated="LONG")) == "LONG"
    assert lps._candidate_side(_sig(SignalAction.WAIT)) is None                    # sıradan WAIT
    assert lps._candidate_side(_sig(SignalAction.WAIT, meta_p_win=0.3)) is None    # taraf bilinmiyor


def test_neural_shadow_pwin_sarmalayici(monkeypatch, stub_shadow):
    monkeypatch.setattr(ns, "get_shadow", lambda: stub_shadow)
    out = lps._neural_shadow_pwin(_feats(), "LONG")
    assert out == {"pwin": 0.7, "std": 0.05, "side": "LONG"}

    def boom():
        raise RuntimeError("yükleme çöktü")
    monkeypatch.setattr(ns, "get_shadow", boom)
    assert lps._neural_shadow_pwin(_feats(), "LONG") is None   # fail-soft


def test_get_shadow_tekil_ve_reload(monkeypatch):
    monkeypatch.setenv("NEURAL_SHADOW", "off")
    ns.reload_shadow()
    a = ns.get_shadow()
    assert ns.get_shadow() is a          # tekil
    ns.reload_shadow()
    assert ns.get_shadow() is not a      # tazelendi
    ns.reload_shadow()
