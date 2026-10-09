"""Config doğrulama testleri — geçersiz .env değerleri yakalanmalı."""
import nihai.config as config


def test_default_config_valid():
    # Varsayılan/mevcut config hatasız olmalı (warning olabilir)
    errors, _ = config.validate_config()
    assert errors == [], f"varsayılan config hata verdi: {errors}"


def test_invalid_sort_caught(monkeypatch):
    monkeypatch.setattr(config, "AUTO_SCREEN_SORT", "expectancyy")  # typo
    errors, _ = config.validate_config()
    assert any("AUTO_SCREEN_SORT" in e for e in errors)


def test_invalid_provider_caught(monkeypatch):
    monkeypatch.setattr(config, "DATA_PROVIDER", "polygon")
    errors, _ = config.validate_config()
    assert any("DATA_PROVIDER" in e for e in errors)


def test_invalid_interval_caught(monkeypatch):
    monkeypatch.setattr(config, "AUTO_SCREEN_INTERVAL", "2h")  # desteklenmiyor
    errors, _ = config.validate_config()
    assert any("AUTO_SCREEN_INTERVAL" in e for e in errors)


def test_negative_equity_caught(monkeypatch):
    monkeypatch.setattr(config, "INITIAL_EQUITY", -100.0)
    errors, _ = config.validate_config()
    assert any("INITIAL_EQUITY" in e for e in errors)


def test_dual_tier_provider_validated_only_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "FOCUSED_PROVIDER", "bogus")
    # dual kapalı → FOCUSED_PROVIDER doğrulanmaz
    monkeypatch.setattr(config, "DUAL_TIER_ENABLED", False)
    errors, _ = config.validate_config()
    assert not any("FOCUSED_PROVIDER" in e for e in errors)
    # dual açık → yakalanır
    monkeypatch.setattr(config, "DUAL_TIER_ENABLED", True)
    errors, _ = config.validate_config()
    assert any("FOCUSED_PROVIDER" in e for e in errors)


def test_same_provider_warns_in_dual(monkeypatch):
    monkeypatch.setattr(config, "DUAL_TIER_ENABLED", True)
    monkeypatch.setattr(config, "FOCUSED_PROVIDER", "yfinance")
    monkeypatch.setattr(config, "BROAD_PROVIDER", "yfinance")
    errors, warnings = config.validate_config()
    assert any("FOCUSED_PROVIDER == BROAD_PROVIDER" in w for w in warnings)


def test_ruin_filter_without_mc_warns(monkeypatch):
    monkeypatch.setattr(config, "AUTO_SCREEN_MAX_RUIN", 0.1)
    monkeypatch.setattr(config, "AUTO_SCREEN_MC", 0)
    _, warnings = config.validate_config()
    assert any("AUTO_SCREEN_MAX_RUIN" in w for w in warnings)


def test_regime_negative_mult_caught():
    # A4 — rejim açıkken negatif çarpan hata vermeli
    reg = config.CONFIG.regime
    saved = (reg.enable_regime_risk, reg.defensive_mult)
    object.__setattr__(reg, "enable_regime_risk", True)
    object.__setattr__(reg, "defensive_mult", -0.5)
    try:
        errors, _ = config.validate_config()
        assert any("REGIME_DEFENSIVE_MULT" in e for e in errors)
    finally:
        object.__setattr__(reg, "enable_regime_risk", saved[0])
        object.__setattr__(reg, "defensive_mult", saved[1])


def test_hybrid_rs_horizon_order_warning(monkeypatch):
    # B8 — kısa ufuk ≥ uzun ufuk → uyarı
    monkeypatch.setattr(config, "SCREEN_HYBRID_ENABLED", True)
    monkeypatch.setattr(config, "SCREEN_RS_SHORT_DAYS", 90)
    monkeypatch.setattr(config, "SCREEN_RS_LONG_DAYS", 21)
    _, warnings = config.validate_config()
    assert any("RS_SHORT_DAYS" in w for w in warnings)


def test_mfe_giveback_nonpositive_caught():
    # A2 — give-back açıkken pozitif olmayan R hata vermeli
    r = config.CONFIG.risk
    saved = (r.enable_mfe_giveback, r.mfe_giveback_r)
    object.__setattr__(r, "enable_mfe_giveback", True)
    object.__setattr__(r, "mfe_giveback_r", 0.0)
    try:
        errors, _ = config.validate_config()
        assert any("MFE_GIVEBACK" in e for e in errors)
    finally:
        object.__setattr__(r, "enable_mfe_giveback", saved[0])
        object.__setattr__(r, "mfe_giveback_r", saved[1])
