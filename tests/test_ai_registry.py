"""registry.py testleri — model yaşam döngüsü/kart + gölge A/B (Tur-16 WP-O kalan kısım)."""
from __future__ import annotations

import json
import random

import pytest

from nihai.ai import registry


@pytest.fixture()
def base(tmp_path):
    """Her test kendi izole kayıt kökünü kullanır (gerçek data/'ya ASLA yazılmaz)."""
    return tmp_path / "ai_models"


def _meta(tier="mlp", **extra):
    m = {"tier": tier, "n_features": 25, "param_count": 3777, "oos_auc": 0.55,
        "dsr": 0.10, "training_env": "pytest"}
    m.update(extra)
    return m


# ---------------------------------------------------------------------------
# register / içerik-hash
# ---------------------------------------------------------------------------
def test_register_kart_alanlari(base):
    card = registry.register(_meta(), artifacts_base_path=base)
    assert "hash" in card and "registered_at" in card
    assert card["tier"] == "mlp"
    assert (base / card["hash"] / "card.json").exists()


def test_register_ayni_icerik_ayni_hash(base):
    c1 = registry.register(_meta(oos_auc=0.55), artifacts_base_path=base)
    c2 = registry.register(_meta(oos_auc=0.55), artifacts_base_path=base)
    assert c1["hash"] == c2["hash"]


# ---------------------------------------------------------------------------
# deprecate / undeprecate — B-5 (bozuk ölçümlü kart seçime girmemeli)
# ---------------------------------------------------------------------------
def test_deprecate_karti_silmez_bayrak_ekler(base):
    card = registry.register(_meta(dsr=1.0), artifacts_base_path=base)
    out = registry.deprecate(card["hash"], "ACI DERS #8 artefaktı", artifacts_base_path=base)
    assert out is not None and out["deprecated"]["reason"] == "ACI DERS #8 artefaktı"
    assert out["deprecated"]["at"]
    # kart dosyası DURUYOR ve ölçüm alanları değişmemiş (bilimsel kayıt korunur)
    disk = json.loads((base / card["hash"] / "card.json").read_text(encoding="utf-8"))
    assert disk["dsr"] == 1.0 and disk["hash"] == card["hash"]
    assert registry.get_card(card["hash"], artifacts_base_path=base) is not None


def test_list_models_emekli_karti_varsayilan_olarak_eler(base):
    iyi = registry.register(_meta(oos_auc=0.52), artifacts_base_path=base)
    bozuk = registry.register(_meta(oos_auc=0.50, dsr=1.0), artifacts_base_path=base)
    registry.deprecate(bozuk["hash"], "kapı bozuk", artifacts_base_path=base)

    guncel = {c["hash"] for c in registry.list_models(artifacts_base_path=base)}
    assert guncel == {iyi["hash"]}
    tumu = {c["hash"] for c in registry.list_models(artifacts_base_path=base,
                                                   include_deprecated=True)}
    assert tumu == {iyi["hash"], bozuk["hash"]}


def _set_registered_at(base, card_hash, ts):
    """`registered_at` saniye çözünürlüklü — aynı saniyede kaydedilen kartlar berabere
    kalır. Kronoloji testinin saat hassasiyetine bağlı olmaması için damgayı sabitle."""
    p = base / card_hash / "card.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["registered_at"] = ts
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def test_latest_emekli_karti_secmez(base):
    eski = registry.register(_meta(tier="mlp", oos_auc=0.52), artifacts_base_path=base)
    yeni = registry.register(_meta(tier="mlp", oos_auc=0.99, dsr=1.0), artifacts_base_path=base)
    _set_registered_at(base, eski["hash"], "2026-01-01T00:00:00+00:00")
    _set_registered_at(base, yeni["hash"], "2026-06-01T00:00:00+00:00")
    # `yeni` daha sonra kaydedildiği için normalde latest() onu döndürürdü
    assert registry.latest(artifacts_base_path=base)["hash"] == yeni["hash"]
    registry.deprecate(yeni["hash"], "sınıf-dengesizliği artefaktı", artifacts_base_path=base)
    assert registry.latest(artifacts_base_path=base)["hash"] == eski["hash"]
    assert registry.latest(artifacts_base_path=base,
                           include_deprecated=True)["hash"] == yeni["hash"]


def test_register_ayni_icerikle_tekrar_emekliligi_korur(base):
    """Aynı eğitimin tekrar koşulması bir küratör kararını sessizce geri almamalı."""
    card = registry.register(_meta(dsr=1.0), artifacts_base_path=base)
    registry.deprecate(card["hash"], "bozuk kapı", artifacts_base_path=base)
    tekrar = registry.register(_meta(dsr=1.0), artifacts_base_path=base)
    assert tekrar["hash"] == card["hash"]
    assert tekrar.get("deprecated", {}).get("reason") == "bozuk kapı"
    assert registry.list_models(artifacts_base_path=base) == []


def test_undeprecate_bayragi_kaldirir_idempotent(base):
    card = registry.register(_meta(), artifacts_base_path=base)
    registry.deprecate(card["hash"], "yanlışlıkla", artifacts_base_path=base)
    geri = registry.undeprecate(card["hash"], artifacts_base_path=base)
    assert "deprecated" not in geri
    assert len(registry.list_models(artifacts_base_path=base)) == 1
    # ikinci çağrı da güvenli (idempotent)
    assert "deprecated" not in registry.undeprecate(card["hash"], artifacts_base_path=base)


def test_deprecate_gerekcesiz_reddeder(base):
    card = registry.register(_meta(), artifacts_base_path=base)
    with pytest.raises(ValueError):
        registry.deprecate(card["hash"], "   ", artifacts_base_path=base)


def test_deprecate_olmayan_kart_none(base):
    base.mkdir(parents=True, exist_ok=True)
    assert registry.deprecate("yok-boyle-bir-hash", "sebep", artifacts_base_path=base) is None
    assert registry.undeprecate("yok-boyle-bir-hash", artifacts_base_path=base) is None


def test_is_deprecated_savunmaci():
    assert registry.is_deprecated(None) is False
    assert registry.is_deprecated({}) is False
    assert registry.is_deprecated({"deprecated": {"at": "x", "reason": "y"}}) is True


def test_register_farkli_icerik_farkli_hash(base):
    c1 = registry.register(_meta(oos_auc=0.55), artifacts_base_path=base)
    c2 = registry.register(_meta(oos_auc=0.60), artifacts_base_path=base)
    assert c1["hash"] != c2["hash"]


def test_register_dict_olmayan_girdi_value_error(base):
    with pytest.raises(ValueError):
        registry.register(["not", "a", "dict"], artifacts_base_path=base)  # type: ignore[arg-type]


def test_register_uretim_dosyalarina_dokunmaz(base, tmp_path):
    """Anayasa: register data/meta_model.* dosyalarına ASLA yazmaz — yalnız base altına yazar."""
    registry.register(_meta(), artifacts_base_path=base)
    assert not (tmp_path / "meta_model.json").exists()
    assert not (tmp_path / "meta_model.pkl").exists()


# ---------------------------------------------------------------------------
# get_card / list_models — savunmacı yükleme
# ---------------------------------------------------------------------------
def test_get_card_bulur(base):
    card = registry.register(_meta(), artifacts_base_path=base)
    fetched = registry.get_card(card["hash"], artifacts_base_path=base)
    assert fetched == card


def test_get_card_bilinmeyen_hash_none(base):
    assert registry.get_card("deadbeef" * 8, artifacts_base_path=base) is None


def test_get_card_bozuk_json_none(base):
    d = base / "bozukhash"
    d.mkdir(parents=True)
    (d / "card.json").write_text("{ gecersiz json", encoding="utf-8")
    assert registry.get_card("bozukhash", artifacts_base_path=base) is None


def test_list_models_bos_kok_bos_liste(base):
    assert registry.list_models(artifacts_base_path=base) == []


def test_list_models_tumunu_dondurur(base):
    c1 = registry.register(_meta(tier="mlp"), artifacts_base_path=base)
    c2 = registry.register(_meta(tier="gbm_small"), artifacts_base_path=base)
    hashes = {c["hash"] for c in registry.list_models(artifacts_base_path=base)}
    assert hashes == {c1["hash"], c2["hash"]}


def test_list_models_bozuk_karti_atlar(base):
    registry.register(_meta(), artifacts_base_path=base)
    bad = base / "bozuk"
    bad.mkdir(parents=True)
    (bad / "card.json").write_text("not json at all", encoding="utf-8")
    cards = registry.list_models(artifacts_base_path=base)
    assert len(cards) == 1


# ---------------------------------------------------------------------------
# latest
# ---------------------------------------------------------------------------
def test_latest_tier_filtresi(base):
    c_mlp = registry.register(_meta(tier="mlp"), artifacts_base_path=base)
    registry.register(_meta(tier="gbm_small"), artifacts_base_path=base)
    got = registry.latest(tier="mlp", artifacts_base_path=base)
    assert got is not None and got["hash"] == c_mlp["hash"]


def test_latest_eslesmeyen_tier_none(base):
    registry.register(_meta(tier="mlp"), artifacts_base_path=base)
    assert registry.latest(tier="quant_transformer", artifacts_base_path=base) is None


def test_latest_bos_kok_none(base):
    assert registry.latest(artifacts_base_path=base) is None


def test_latest_en_son_registered_at(base, monkeypatch):
    """registered_at'ı en büyük (en son) kayıt döner — registered_at sırasını sahteleyerek doğrula."""
    c_old = registry.register(_meta(tier="mlp", tag="eski"), artifacts_base_path=base)
    c_new = registry.register(_meta(tier="mlp", tag="yeni"), artifacts_base_path=base)
    # registered_at'ı manuel olarak sırala (aynı saniyede iki çağrı eşitlenebilir).
    old_card = registry.get_card(c_old["hash"], artifacts_base_path=base)
    old_card["registered_at"] = "2000-01-01T00:00:00+00:00"
    (base / c_old["hash"] / "card.json").write_text(json.dumps(old_card), encoding="utf-8")
    new_card = registry.get_card(c_new["hash"], artifacts_base_path=base)
    new_card["registered_at"] = "2999-01-01T00:00:00+00:00"
    (base / c_new["hash"] / "card.json").write_text(json.dumps(new_card), encoding="utf-8")
    got = registry.latest(tier="mlp", artifacts_base_path=base)
    assert got["hash"] == c_new["hash"]


# ---------------------------------------------------------------------------
# shadow_ab_log — atomik JSONL
# ---------------------------------------------------------------------------
def test_shadow_ab_log_satir_yazar(base):
    row = registry.shadow_ab_log("hashA", "hashB", "AAPL", 0.6, 0.4, outcome=1, artifacts_base_path=base)
    assert row["model_a"] == "hashA" and row["symbol"] == "AAPL"
    path = base / registry.AB_LOG_FILENAME
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["p_a"] == 0.6 and parsed["outcome"] == 1


def test_shadow_ab_log_coklu_ekleme_tum_satirlar_gecerli_json(base):
    for i in range(20):
        registry.shadow_ab_log("hashA", "hashB", f"SYM{i}", 0.5, 0.5, outcome=i % 2,
                               artifacts_base_path=base)
    path = base / registry.AB_LOG_FILENAME
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    for line in lines:
        json.loads(line)  # her satır bağımsız geçerli JSON olmalı


def test_shadow_ab_log_outcome_yoksa_none(base):
    row = registry.shadow_ab_log("hashA", "hashB", "AAPL", 0.6, 0.4, artifacts_base_path=base)
    assert row["outcome"] is None


# ---------------------------------------------------------------------------
# ab_report — istatistiksel karşılaştırma
# ---------------------------------------------------------------------------
def _fill_ab_log(base, a_hash, b_hash, n, skill_a=0.1, skill_b=0.1, seed=0):
    """n gözlemli sentetik A/B log: outcome~Bernoulli(0.5); p_x = 0.5 + skill_x·(±1 doğru
    yöne) + GENİŞ gürültü (U(-0.5,0.5) — skill'in TEK BAŞINA kararı belirlemesini önler,
    aksi halde isabet hep %100/%0'a doygunlaşır ve iki modeli AYIRT ETMEK imkansız olur).
    Beklenen isabet ≈ 0.5+skill (kapalı-form: bkz. yorum) — `skill` ne kadar büyükse
    tahmin o kadar isabetli (0≈yazı-tura, 0.5≈kusursuz)."""
    rng = random.Random(seed)
    for i in range(n):
        y = 1 if rng.random() < 0.5 else 0
        sign = 1.0 if y == 1 else -1.0
        p_a = min(0.999, max(0.001, 0.5 + sign * skill_a + rng.uniform(-0.5, 0.5)))
        p_b = min(0.999, max(0.001, 0.5 + sign * skill_b + rng.uniform(-0.5, 0.5)))
        registry.shadow_ab_log(a_hash, b_hash, f"SYM{i}", p_a, p_b, outcome=y, artifacts_base_path=base)


def test_ab_report_yetersiz_ornek(base):
    _fill_ab_log(base, "A", "B", n=10)
    rep = registry.ab_report("A", "B", min_n=100, artifacts_base_path=base)
    assert rep["verdict"] == "YETERSIZ_ORNEK"
    assert rep["n"] == 10
    assert "brier_a" not in rep


def test_ab_report_farksiz_benzer_performans(base):
    """İki model AYNI kaliteliyse (aynı skill) Wilson CI'ları çakışır → FARKSIZ."""
    _fill_ab_log(base, "A", "B", n=300, skill_a=0.1, skill_b=0.1, seed=1)
    rep = registry.ab_report("A", "B", min_n=100, artifacts_base_path=base)
    assert rep["verdict"] == "FARKSIZ"


def test_ab_report_a_ustun_belirgin_fark(base):
    """A modeli sistematik olarak daha isabetliyse (büyük skill farkı) CI'lar ayrışır → A_USTUN."""
    _fill_ab_log(base, "A", "B", n=300, skill_a=0.35, skill_b=0.02, seed=2)
    rep = registry.ab_report("A", "B", min_n=100, artifacts_base_path=base)
    assert rep["verdict"] == "A_USTUN"
    assert rep["isabet_a"] > rep["isabet_b"]
    assert rep["brier_a"] < rep["brier_b"]


def test_ab_report_outcome_yok_satirlar_sayilmaz(base):
    _fill_ab_log(base, "A", "B", n=150, seed=3)
    for i in range(50):
        registry.shadow_ab_log("A", "B", f"NOOUTCOME{i}", 0.5, 0.5, artifacts_base_path=base)
    rep = registry.ab_report("A", "B", min_n=100, artifacts_base_path=base)
    assert rep["n"] == 150  # outcome=None satırlar hariç


def test_ab_report_farkli_ciftler_izole(base):
    """(A,B) log satırları (C,D) çiftinin raporunu ETKİLEMEZ — izolasyon garantisi."""
    _fill_ab_log(base, "A", "B", n=150, skill_a=0.35, skill_b=0.02, seed=4)
    rep_cd = registry.ab_report("C", "D", min_n=1, artifacts_base_path=base)
    assert rep_cd["n"] == 0
    assert rep_cd["verdict"] == "YETERSIZ_ORNEK"


def test_ab_report_bos_log_yetersiz(base):
    rep = registry.ab_report("X", "Y", min_n=1, artifacts_base_path=base)
    assert rep["n"] == 0 and rep["verdict"] == "YETERSIZ_ORNEK"


def test_ab_report_wilson_araliklari_0_1_arasinda(base):
    _fill_ab_log(base, "A", "B", n=200, skill_a=0.35, skill_b=0.15, seed=5)
    rep = registry.ab_report("A", "B", min_n=100, artifacts_base_path=base)
    for key in ("wilson_a", "wilson_b"):
        lo, hi = rep[key]
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Tur-24 optimizasyonu: ab_log.jsonl EKLEME O(satır) olmalı, O(dosya) değil
# ---------------------------------------------------------------------------
def test_ab_log_ekleme_dosyayi_yeniden_yazmaz(base, monkeypatch):
    """`shadow_ab_log` artık her çağrıda tüm dosyayı okuyup yeniden YAZMAMALI.
    `atomic_write_text` bu yolda hiç çağrılmazsa (kart yazımları ayrı yoldur)
    O(dosya) terimi gerçekten kalkmış demektir."""
    cagrildi: list[str] = []
    monkeypatch.setattr(registry, "atomic_write_text",
                        lambda p, t: cagrildi.append(str(p)))
    for i in range(5):
        registry.shadow_ab_log("A", "B", f"SYM{i}", 0.6, 0.4, outcome=1,
                               artifacts_base_path=base)
    assert cagrildi == []
    rep = registry.ab_report("A", "B", min_n=1, artifacts_base_path=base)
    assert rep["n"] == 5


def test_ab_log_satirlari_ekleme_sirasinda_birikir(base):
    """Ekleme gerçekten APPEND: önceki satırlar korunur, sıra bozulmaz."""
    for i in range(4):
        registry.shadow_ab_log("A", "B", f"SYM{i}", 0.5 + i / 100, 0.5, outcome=i % 2,
                               artifacts_base_path=base)
    lines = (base / registry.AB_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert [json.loads(ln)["symbol"] for ln in lines] == ["SYM0", "SYM1", "SYM2", "SYM3"]


def test_yarim_kalan_son_satir_raporu_bozmaz(base):
    """Dayanıklılık sözleşmesi: çökme SON satırı yarım bırakabilir; okuyucu onu atlar
    ve önceki gözlemler KAYBOLMAZ (tam-yeniden-yazımın aksine tüm dosya riske girmez)."""
    for i in range(3):
        registry.shadow_ab_log("A", "B", f"SYM{i}", 0.6, 0.4, outcome=1,
                               artifacts_base_path=base)
    p = base / registry.AB_LOG_FILENAME
    with p.open("a", encoding="utf-8") as f:
        f.write('{"ts": "2026-08-10", "model_a": "A", "mod')   # yarım satır (fsync öncesi çökme)
    rep = registry.ab_report("A", "B", min_n=1, artifacts_base_path=base)
    assert rep["n"] == 3
