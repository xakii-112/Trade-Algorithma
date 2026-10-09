"""BULGU-1 kalıcı regresyon guard'ı — MainHub/ ve nihai/data/ altındaki iki
alpaca_stream_provider.py kopyasının kritik sağlamlık/veri-kalite özelliklerini
KORUDUĞUNU doğrular. İki dosya yalnız import yoluyla farklı olmalı; biri diğerinden
geri kalırsa (eskiden olduğu gibi) bu test kırılır.

MainHub yoksa (nihai tek başına checkout) test atlanır."""
from pathlib import Path

import pytest

_NIHAI = Path(__file__).resolve().parents[1] / "nihai" / "data" / "alpaca_stream_provider.py"
_MAINHUB = Path(__file__).resolve().parents[2] / "MainHub" / "alpaca_stream_provider.py"

# Her iki kopyada da BULUNMASI gereken kritik kod imzaları (birleşim/union özellikleri):
_REQUIRED = [
    "_patched_run_forever",                       # CPU-spin/log-flood koruması (monkeypatch)
    "DataStream._run_forever = _patched_run_forever",
    "subscribe_trades",                           # gerçek son-işlem aboneliği
    "async def _on_trade",                        # gerçek son-işlem işleyicisi
    "hit[2] >= lookback_bars",                    # lookback-duyarlı seed cache
    "symbols_with_trades",                        # stream_status trade sayacı
    "backoff = min(backoff * 2.0, 60.0)",         # üstel reconnect backoff
]


@pytest.mark.skipif(not _MAINHUB.exists(), reason="MainHub kopyası yok (tek-başına checkout)")
@pytest.mark.parametrize("path", [_NIHAI, _MAINHUB], ids=["nihai", "mainhub"])
def test_stream_provider_retains_reconciled_features(path):
    src = path.read_text(encoding="utf-8")
    missing = [sig for sig in _REQUIRED if sig not in src]
    assert not missing, f"{path.name} şu birleştirilmiş özellikleri kaybetmiş: {missing}"


@pytest.mark.skipif(not _MAINHUB.exists(), reason="MainHub kopyası yok")
def test_two_copies_differ_only_by_import_paths():
    """İki dosya, import yolları (nihai.data/nihai.config ↔ top-level) ve docstring
    NOT'u dışında ÖZDEŞ gövdeye sahip olmalı — sınıf/metot kodu birebir aynı."""
    def normalize(p: Path) -> str:
        out = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.replace("\r", "")
            # import yolu farklarını eşitle
            s = (s.replace("from nihai.data.alpaca_provider", "from alpaca_provider")
                  .replace("from nihai.data.data_provider", "from data_provider")
                  .replace("from nihai.config", "from config"))
            # yalnız import satırlarını + boş/yorum/docstring gürültüsünü tutmuyoruz:
            # gövde (def/return/atama vs.) kıyaslaması için kod satırlarını al
            stripped = s.strip()
            if not stripped or stripped.startswith("#"):
                continue
            out.append(s)
        return "\n".join(out)

    # Sınıf gövdesini kıyasla: ilk "class AlpacaStreamProvider" sonrası kod.
    def class_body(text: str) -> str:
        marker = "class AlpacaStreamProvider"
        idx = text.find(marker)
        return text[idx:] if idx >= 0 else text

    nihai_body = class_body(normalize(_NIHAI))
    main_body = class_body(normalize(_MAINHUB))
    assert nihai_body == main_body, "İki stream sağlayıcı sınıf gövdesi ıraksamış (import dışı fark)"
