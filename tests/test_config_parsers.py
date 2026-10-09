"""config.py saf parse yardımcılarının (_parse_tiers, _envb) edge-case testleri.

Bunlar .env girdisini yorumlayan davranış-kritik fonksiyonlar; bozuk girdi
sessizce göz ardı edilmeli ve tiered-trail sıralaması (en yüksek kâr eşiği önce)
korunmalı — yoksa trailing-stop yanlış basamağı seçer."""
import nihai.config as config

# ---------------- _parse_tiers ----------------

def test_parse_tiers_empty_returns_empty():
    assert config._parse_tiers("") == ()


def test_parse_tiers_none_returns_empty():
    # `raw or ""` koruması None'da çökmemeli
    assert config._parse_tiers(None) == ()  # type: ignore[arg-type]


def test_parse_tiers_single_pair():
    assert config._parse_tiers("1.5:2.0") == ((1.5, 2.0),)


def test_parse_tiers_sorted_descending_by_threshold():
    # Girdi karışık sırada; çıktı kâr eşiğine göre AZALAN olmalı (lookup ilk eşleşmeyi alır)
    out = config._parse_tiers("1.5:2.0,5.0:1.0,3.0:1.5")
    assert out == ((5.0, 1.0), (3.0, 1.5), (1.5, 2.0))
    assert [t[0] for t in out] == sorted([t[0] for t in out], reverse=True)


def test_parse_tiers_skips_malformed_chunks():
    # ':' yok, float değil, fazla alan → atlanır; geçerli olan kalır
    out = config._parse_tiers("abc,1.5:2.0,3.0,x:y,4.0:1.0:9")
    assert out == ((1.5, 2.0),)


def test_parse_tiers_ignores_blank_and_whitespace():
    out = config._parse_tiers("  , 2.0:1.0 ,, 4.0:0.5 ")
    assert out == ((4.0, 0.5), (2.0, 1.0))


def test_parse_tiers_all_garbage_returns_empty():
    assert config._parse_tiers("foo,bar:baz,::") == ()


# ---------------- _envb ----------------

def test_envb_unset_returns_default(monkeypatch):
    monkeypatch.delenv("SOME_FLAG_XYZ", raising=False)
    assert config._envb("SOME_FLAG_XYZ", True) is True
    assert config._envb("SOME_FLAG_XYZ", False) is False


def test_envb_truthy_tokens(monkeypatch):
    for tok in ("1", "true", "TRUE", "Yes", "on", "  on  "):
        monkeypatch.setenv("SOME_FLAG_XYZ", tok)
        assert config._envb("SOME_FLAG_XYZ", False) is True, tok


def test_envb_falsy_tokens(monkeypatch):
    for tok in ("0", "false", "no", "off", "", "maybe"):
        monkeypatch.setenv("SOME_FLAG_XYZ", tok)
        assert config._envb("SOME_FLAG_XYZ", True) is False, tok
